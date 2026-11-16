"""Safe wrapper around ur_rtde for URSim / a real UR5e.

    with Robot() as robot:          # connects to 127.0.0.1, disconnects on exit
        robot.move_j([0, -1.57, 1.57, 0, 1.57, 0])
        robot.move_l([-0.45, 0.0, 0.30] + TOOL_DOWN)

Every move is checked first (robot state, speed/accel, joint limits, workspace box, controller
safety limits) and verified afterwards; it returns True on success and False with a printed reason
otherwise. Only one RTDEControlInterface can be connected at a time: close it before reconnecting.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_IP = "127.0.0.1"
TOOL_DOWN = [0.0, math.pi, 0.0]   # rotation vector: tool Z axis pointing down at the table
ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


@dataclass
class Limits:
    joint_limit: float = math.pi        # rad, |q_i| allowed (controller allows 2π)
    max_joint_speed: float = 1.0        # rad/s
    max_joint_accel: float = 1.4        # rad/s²
    joint_tolerance: float = 1e-3       # rad, accepted final joint error
    max_lin_speed: float = 0.25         # m/s
    max_lin_accel: float = 0.5          # m/s²
    position_tolerance: float = 0.002   # m, accepted final TCP error
    workspace: dict = field(default_factory=lambda: {   # m, allowed TCP box (base frame)
        "x": (-0.80, -0.15), "y": (-0.60, 0.60), "z": (0.01, 0.60)})


class Robot:
    def __init__(self, ip=DEFAULT_IP, limits=None, gripper_do=0):
        from rtde_control import RTDEControlInterface
        from rtde_io import RTDEIOInterface
        from rtde_receive import RTDEReceiveInterface

        self.limits = limits or Limits()
        self.gripper_do = gripper_do      # tool digital output used as the gripper
        self.receive = RTDEReceiveInterface(ip)
        self.control = RTDEControlInterface(ip)
        self.io = RTDEIOInterface(ip)

    # --- connection -------------------------------------------------------------------------
    def close(self):
        try:
            self.control.stopScript()
        finally:
            for iface in (self.control, self.io, self.receive):
                try:
                    iface.disconnect()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- state ------------------------------------------------------------------------------
    def joints(self):
        return self.receive.getActualQ()

    def tcp_pose(self):
        return self.receive.getActualTCPPose()

    def is_ready(self):
        r = self.receive
        return (not r.isProtectiveStopped() and not r.isEmergencyStopped()
                and r.getRobotMode() == ROBOT_MODE_RUNNING and r.getSafetyMode() == SAFETY_MODE_NORMAL)

    @staticmethod
    def _refuse(reason):
        print(f"❌ {reason}")
        return False

    # --- motion -----------------------------------------------------------------------------
    def move_j(self, q, speed=0.3, accel=0.5):
        """Joint move to q (6 angles, rad)."""
        lim = self.limits
        if len(q) != 6 or not all(math.isfinite(v) for v in q):
            return self._refuse("target must be 6 finite joint values")
        if not (0 < speed <= lim.max_joint_speed and 0 < accel <= lim.max_joint_accel):
            return self._refuse(f"speed/accel out of range: {speed}, {accel}")
        for i, v in enumerate(q):
            if abs(v) > lim.joint_limit:
                return self._refuse(f"joint {i} = {v:.3f} rad outside ±{lim.joint_limit:.3f}")
        if not self.is_ready():
            return self._refuse("robot not ready (stopped, or not RUNNING/NORMAL)")
        ok = self.control.moveJ(list(q), speed, accel)
        error = max(abs(a - t) for a, t in zip(self.joints(), q))
        if not ok or error > lim.joint_tolerance:
            return self._refuse(f"moveJ failed (returned {ok}, joint error {error:.4f} rad)")
        return True

    def move_l(self, pose, speed=0.1, accel=0.3):
        """Linear move to pose [x, y, z, rx, ry, rz] (base frame, rotation vector)."""
        lim = self.limits
        if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
            return self._refuse("pose must be 6 finite numbers")
        if not (0 < speed <= lim.max_lin_speed and 0 < accel <= lim.max_lin_accel):
            return self._refuse(f"speed/accel out of range: {speed}, {accel}")
        for value, axis in zip(pose[:3], "xyz"):
            low, high = lim.workspace[axis]
            if not low <= value <= high:
                return self._refuse(f"{axis}={value:.3f} m outside workspace [{low}, {high}]")
        if not self.is_ready():
            return self._refuse("robot not ready (stopped, or not RUNNING/NORMAL)")
        if not self.control.isPoseWithinSafetyLimits(list(pose)):
            return self._refuse("pose violates the controller safety limits")
        if not self.control.moveL(list(pose), speed, accel):
            return self._refuse("moveL failed")
        error = float(np.linalg.norm(np.array(self.tcp_pose()[:3]) - np.array(pose[:3])))
        if error > lim.position_tolerance:
            return self._refuse(f"position error {error * 1000:.1f} mm")
        return True

    def gripper(self, close, wait=0.5):
        """Close (True) or open (False) the simulated gripper (a tool digital output)."""
        self.io.setToolDigitalOut(self.gripper_do, close)
        time.sleep(wait)
        return self.receive.getDigitalOutState(16 + self.gripper_do) == close   # tool DOs are bits 16-17
