"""Make the UR5 follow a coloured object with an eye-in-hand camera (image-based visual servoing).

The camera is mounted on the tool and looks along the tool Z axis (down at the table). Each frame:
  1. detect the object by colour (HSV, same pipeline as Day 2). With --shape, keep only the blob
     whose silhouette matches a reference photo (scale and rotation do not matter),
  2. measure its offset from the image centre,
  3. turn that offset into a TCP velocity in the robot base frame (proportional control),
  4. integrate the velocity into a target pose and stream it with servoL at 50 Hz.
The TCP keeps its orientation and moves in a plane at right angles to the camera's view: base X/Y
for a camera looking down, or, with --start-here, the start pose's tool X/Y (a camera looking
forward then moves sideways and up/down, never toward the object).

Modes
  --sim            Synthetic camera rendered from URSim's actual TCP pose, with a virtual coloured
                   disc moving on the table. The loop is really closed, so use this with URSim.
                   Click in the window to drop the object there, space pauses its motion.
  (default)        Real webcam (--camera N). Only closes the loop on the REAL robot with the camera
                   on the tool: in URSim the image does not change when the robot moves, so the robot
                   keeps moving toward the object until it reaches the workspace limit.
  --webcam-table   PC webcam in URSim: the webcam image is laid flat on the virtual table and the
                   tool camera sees the part under the TCP. Move a coloured object in front of the
                   webcam and the robot goes after it. Inset: webcam + tool camera footprint.
  --start-here     Real robot: jog the camera above the table with the pendant first. Follows at the
                   current height and orientation, inside a +-RADIUS box around the current XY.
  --calibrate      Before following, move +-5 cm along both plane axes and measure how a still object
                   moves in the image (image Jacobian), then save it. Use it on the real robot: no need
                   to know how the camera is mounted. --load-calib reuses the saved one (no moves).
  --no-robot       Detection and computed velocity only, no robot connection (tune HSV ranges here).
  --hand           Follow a hand instead of a colour (MediaPipe): a hand is followed whether it is
                   open or closed; no hand sends the robot back to where following started (the --home
                   pose with --home), at --max-speed. Works with every camera mode except --sim.
                   With --gripper the claw copies the hand: a fist closes it, an open hand opens it
                   (losing the hand leaves it as it is).
  --shape [PHOTO]  Also match the object's silhouette. PHOTO is a picture where the object is the
                   largest blob of --color (default: the newest image in data/). A blue circle or
                   a long bar is ignored; a blob the same shape as the photo is kept.

Examples (URSim running, see ursim/bringup.sh):
    uv run python project/follow_object.py --sim
    uv run python project/follow_object.py --sim --headless --duration 30   # self-test, prints error
    uv run python project/follow_object.py --webcam-table --color red   # PC webcam + URSim
    uv run python project/follow_object.py --no-robot --color blue      # webcam, no robot
    uv run python project/follow_object.py --no-robot --color blue --camera realsense
    uv run python project/follow_object.py --no-robot --hand            # hand detection only
See README.md (repository root) for every option and parameter.
Real robot (camera on the tool, object visible and still, hand on the e-stop):
    uv run python project/follow_object.py --ip <ROBOT_IP> --camera realsense --start-here --calibrate \
        --max-speed 0.02 --gain 1.5 --color red
Keys: q / Esc quit, space pause (sim object motion), click: move the sim object,
      s learn the shape from the current frame (saved to data/, like --shape), x forget the shape.
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')   # OpenCV's Qt only ships the X11 plugin
import cv2
import numpy as np

from camera import DEPTH_MAX_M, DEPTH_MIN_M, Webcam, open_camera
from cobot.gripper import AsyncGripper, RobotiqGripper, ToolDOGripper
from hand import GRIP_CONFIRM, MODEL_PATH as HAND_MODEL, AsyncHandDetector, HandDetector, HandGate, draw_hand
from detection import COLOR_RANGES, DRAW_BGR, capture_shape, detect, learn_shape, resolve_shape_photo

# ------------------------------------------------------------------ configuration
ROBOT_IP = '127.0.0.1'
TOOL_DOWN = [0.0, math.pi, 0.0]        # rotation vector: tool Z pointing down at the table
WORKSPACE = {'x': (-0.75, -0.20), 'y': (-0.45, 0.45)}   # m, the target is clamped to this box
FOLLOW_Z = 0.30                         # m, TCP height while following
START_XY = (-0.45, 0.0)                 # m, where tracking starts
# Elbow-up, tool-down joint configuration (deg) used as the IK seed for the start pose; every joint
# is far from its limits. Starting with moveJ there avoids winding the wrist from an odd pose.
HOME_SEED_DEG = [-162.8, -88.1, -120.5, -61.4, 90.0, -72.8]
OBJECT_Z = 0.0                          # m, height of the object's top surface (table = 0)

DT = 0.02                               # s, control period (50 Hz)
GAIN = 1.0                              # 1/s, proportional gain: v = GAIN * offset (+ feedforward)
FEEDFORWARD = 0.3                       # 0..1, add the estimated object velocity (0 = pure P control)
VEL_FILTER = 0.15                       # 0..1, low-pass on the object velocity estimate
MAX_SPEED = 0.02                        # m/s, TCP speed cap (kept low for safety; also used for calibration moves)
MAX_ACCEL = 0.15                        # m/s², limit on velocity changes
DEADBAND_PX = 4                         # px, ignore smaller errors (detection jitter)
ERROR_FILTER = 0.5                      # 0..1, low-pass on the measured offset (1 = no filter)
HOME_GAIN = 1.0                         # 1/s, --hand: speed back home = HOME_GAIN x distance (capped)
LOST_TIMEOUT = 0.5                      # s without detection before the robot stops

CAM_HFOV_DEG = 60.0                     # horizontal field of view, for cameras that don't report their intrinsics
CAM_YAW_DEG = 0.0                       # camera rotation about the tool Z axis (mounting)
CAM_FLIP = False                        # True if the image is mirrored relative to the tool frame


def intrinsics(w, h, hfov_deg=CAM_HFOV_DEG):
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def camera_matrix(cam, frame):
    """The camera's own intrinsics if it reports them (RealSense, simulated cameras), else the
    CAM_HFOV_DEG model. A plain webcam can't report its lens through OpenCV."""
    K = getattr(cam, 'K', None)
    h, w = frame.shape[:2]
    if K is not None and abs(K[0, 2] - w / 2) < w / 4 and abs(K[1, 2] - h / 2) < h / 4:
        return np.asarray(K, float)
    return intrinsics(w, h)


def cam_rotation(yaw_deg, flip):
    """Rotation from the camera frame (OpenCV: x right, y down, z forward) to the tool frame."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R @ np.diag([-1, 1, 1]) if flip else R


def tool_to_cam():
    return cam_rotation(CAM_YAW_DEG, CAM_FLIP)


def pose_rotation(pose):
    return cv2.Rodrigues(np.asarray(pose[3:6], dtype=float))[0]


# ------------------------------------------------------------------ control
BASE_XY = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


class Follower:
    """Image offset -> TCP velocity in the servo plane, with deadband, filter, speed and accel limits.

    The servo plane is spanned by the two columns of A (unit vectors in the base frame) through
    `origin`: base X/Y for a camera looking down in URSim, the start pose's tool X/Y with
    --start-here (the plane at right angles to the camera's view, whichever way it looks).
    Velocities and positions below are 2D coordinates in that plane.
    """

    def __init__(self, A=BASE_XY, origin=(0.0, 0.0, 0.0)):
        self.A, self.origin = np.asarray(A, float), np.asarray(origin, float)
        self.err = None          # filtered offset (normalised image coordinates)
        self.v = np.zeros(2)     # current commanded velocity in the plane, m/s
        self.last_seen = -1e9
        self.obj = None          # (time, estimated object position in the plane)
        self.v_obj = np.zeros(2) # filtered object velocity estimate, m/s
        self.J = None            # working image Jacobian (px per m), possibly rescaled by live depth
        self.J_calib = None      # Jacobian at calibration distance
        self.distance_m = None   # object distance (m) when J_calib was measured
        self.depth_now = None    # last live depth (m) used for rescale / overlay
        self.fxy = None          # focal lengths (fx, fy), px

    def set_jacobian(self, J, distance_m=None):
        """Store the calibrated Jacobian and optional reference distance for live depth rescale."""
        self.J_calib = np.asarray(J, float).copy()
        self.J = self.J_calib.copy()
        self.distance_m = None if distance_m is None else float(distance_m)
        self.depth_now = self.distance_m

    def apply_depth(self, d_now):
        """Rescale J so mm/px stays right when the object distance changes: J ∝ 1/d."""
        if self.J_calib is None or d_now is None:
            return
        d_now = float(np.clip(d_now, DEPTH_MIN_M, DEPTH_MAX_M))
        if self.distance_m is None:
            self.distance_m = d_now
        self.J = self.J_calib * (self.distance_m / d_now)
        self.depth_now = d_now

    def offset_base(self, err, tcp_pose):
        """Normalised image offset -> offset (m, plane coordinates) from the camera axis to the object."""
        if self.J is not None:
            # moving the TCP by d shifts the object by J d pixels: it is centred when J d = -offset
            return -np.linalg.solve(self.J, np.asarray(err) * self.fxy)   # back to pixels
        depth = max(tcp_pose[2] - OBJECT_Z, 0.05)
        offset_cam = np.array([err[0] * depth, err[1] * depth, 0.0])
        return self.A.T @ (pose_rotation(tcp_pose) @ tool_to_cam() @ offset_cam)

    def update(self, det, K, tcp_pose, now):
        self.fxy = np.array([K[0, 0], K[1, 1]])
        if det is not None:
            u, v = det[0] - K[0, 2], det[1] - K[1, 2]
            e = np.array([u / K[0, 0], v / K[1, 1]])
            self.err = e if self.err is None else ERROR_FILTER * e + (1 - ERROR_FILTER) * self.err
            # object position = camera axis + offset; its derivative is the object velocity
            obj = self.A.T @ (np.asarray(tcp_pose[:3]) - self.origin) + self.offset_base(e, tcp_pose)
            if self.obj is not None and now - self.obj[0] < LOST_TIMEOUT:
                dt = now - self.obj[0]
                if dt > 1e-3:
                    self.v_obj = VEL_FILTER * (obj - self.obj[1]) / dt + (1 - VEL_FILTER) * self.v_obj
            else:
                self.v_obj = np.zeros(2)
            self.obj = (now, obj)
            self.last_seen = now
        if now - self.last_seen > LOST_TIMEOUT:
            self.err, self.obj = None, None
        if self.err is None:
            v_des = np.zeros(2)  # lost: brake
        else:
            err = self.err if np.linalg.norm(self.err * self.fxy) >= DEADBAND_PX else np.zeros(2)
            v_des = GAIN * self.offset_base(err, tcp_pose) + FEEDFORWARD * self.v_obj
            speed = np.linalg.norm(v_des)
            if speed > MAX_SPEED:
                v_des *= MAX_SPEED / speed
        dv = v_des - self.v
        max_dv = MAX_ACCEL * DT
        if np.linalg.norm(dv) > max_dv:
            dv *= max_dv / np.linalg.norm(dv)
        self.v = self.v + dv
        return self.v

    @property
    def tracking(self):
        return self.err is not None



class DepthFollower:
    """Keep the object at `target` metres from the camera by moving along the camera's view axis.

    Input: depth (m) measured at the detected object, or None. Output: velocity (m/s) along the
    tool Z axis, positive = toward the object. Safe by default: no valid depth for LOST_TIMEOUT
    brakes to zero, and a sudden jump (a hand passing in front, a depth hole) is ignored unless it
    persists for several frames.
    """

    def __init__(self, target, gain, deadband=0.02, smooth=0.3, max_jump=0.15, jump_frames=5):
        self.target, self.gain, self.deadband = target, gain, deadband
        self.smooth, self.max_jump, self.jump_frames = smooth, max_jump, jump_frames
        self.d = None            # filtered object distance, m
        self.v = 0.0             # commanded velocity along tool Z, m/s
        self.last_ok = -1e9
        self.jumps = 0

    def update(self, d_meas, now):
        if d_meas is not None:
            if self.d is None or abs(d_meas - self.d) < self.max_jump:
                self.d = d_meas if self.d is None else self.smooth * d_meas + (1 - self.smooth) * self.d
                self.last_ok, self.jumps = now, 0
            else:
                self.jumps += 1
                if self.jumps >= self.jump_frames:   # a real change of distance, not a glitch
                    self.d, self.last_ok, self.jumps = d_meas, now, 0
        if now - self.last_ok > LOST_TIMEOUT:
            self.d = None
        e = 0.0 if self.d is None else self.d - self.target
        v_des = float(np.clip(self.gain * e, -MAX_SPEED, MAX_SPEED)) if abs(e) > self.deadband else 0.0
        dv = float(np.clip(v_des - self.v, -MAX_ACCEL * DT, MAX_ACCEL * DT))
        self.v += dv
        return self.v


def clamp_xy(x, y):
    return (min(max(x, WORKSPACE['x'][0]), WORKSPACE['x'][1]),
            min(max(y, WORKSPACE['y'][0]), WORKSPACE['y'][1]))


# ------------------------------------------------------------------ cameras
class SimCamera:
    """Renders what a tool camera would see from the robot's actual TCP pose: a table grid and a disc."""

    def __init__(self, color, w=640, h=480, radius=0.03, mount=None):
        self.w, self.h, self.K = w, h, intrinsics(w, h)
        # true mounting (tests): a rotation about the optical axis, plus an optional mirrored image
        yaw, flip = mount if mount else (CAM_YAW_DEG, CAM_FLIP)
        self.R_tc, self.flip = cam_rotation(yaw, False), bool(flip)
        self.color, self.radius = color, radius
        self.center, self.amp, self.period = np.array([-0.47, 0.0]), np.array([0.15, 0.25]), 24.0
        self.t, self.paused, self.manual = 0.0, False, None
        self.last_now, self.seq, self.last_rvec = None, 0, None

    def object_xy(self):
        if self.manual is not None:
            return self.manual
        w = 2 * math.pi / self.period
        return self.center + self.amp * np.array([math.sin(w * self.t), math.sin(2 * w * self.t)])

    def _extrinsics(self, tcp_pose):
        R_bc = pose_rotation(tcp_pose) @ self.R_tc         # camera -> base
        R_cb = R_bc.T
        t_cb = -R_cb @ np.asarray(tcp_pose[:3])
        return R_cb, t_cb

    def project(self, pts, tcp_pose):
        R_cb, t_cb = self._extrinsics(tcp_pose)
        img, _ = cv2.projectPoints(np.asarray(pts, float), cv2.Rodrigues(R_cb)[0], t_cb, self.K, None)
        return img.reshape(-1, 2)

    def pixel_to_table(self, u, v, tcp_pose):
        if self.flip:
            u = self.w - u
        R_cb, t_cb = self._extrinsics(tcp_pose)
        ray = R_cb.T @ np.linalg.solve(self.K, [u, v, 1.0])
        origin = np.asarray(tcp_pose[:3])
        s = (OBJECT_Z - origin[2]) / ray[2]
        return origin[:2] + s * ray[:2]

    def depth_at(self, u, v, window=None):
        """Depth (m, along the optical axis) of the table point seen at pixel (u, v), like a RealSense."""
        pose = getattr(self, 'last_pose', None)
        if pose is None:
            return None
        x, y = self.pixel_to_table(u, v, pose)
        R_cb, t_cb = self._extrinsics(pose)
        z = float((R_cb @ np.array([x, y, OBJECT_Z]) + t_cb)[2])
        return z if z > 0 else None

    def read(self, tcp_pose, now):
        self.last_pose = list(tcp_pose)
        if self.last_now is not None and not self.paused:
            self.t += now - self.last_now
        self.last_now = now
        img = np.full((self.h, self.w, 3), (150, 160, 165), np.uint8)
        for x in np.arange(-0.90, 0.11, 0.05):     # 5 cm grid on the table
            p = self.project([[x, -0.7, 0], [x, 0.7, 0]], tcp_pose)
            cv2.line(img, tuple(map(int, p[0])), tuple(map(int, p[1])), (120, 125, 130), 1, cv2.LINE_AA)
        for y in np.arange(-0.70, 0.71, 0.05):
            p = self.project([[-0.9, y, 0], [0.1, y, 0]], tcp_pose)
            cv2.line(img, tuple(map(int, p[0])), tuple(map(int, p[1])), (120, 125, 130), 1, cv2.LINE_AA)
        ox, oy = self.object_xy()
        a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        disc = np.stack([ox + self.radius * np.cos(a), oy + self.radius * np.sin(a), np.full_like(a, OBJECT_Z)], 1)
        cv2.fillPoly(img, [self.project(disc, tcp_pose).astype(np.int32)], DRAW_BGR[self.color], cv2.LINE_AA)
        if self.flip:
            img = cv2.flip(img, 1)
        img = cv2.add(img, np.random.randint(0, 6, img.shape, np.uint8))   # a little sensor noise
        self.seq += 1
        return img, self.seq

    def close(self):
        pass


class WebcamTable(SimCamera):
    """The PC webcam image laid flat on the table, seen by the virtual tool camera.

    In URSim a real webcam cannot move with the robot, so the eye-in-hand loop would stay open.
    Here the (mirrored) webcam frame is a 'poster' on the table, ZOOM times larger than the tool
    camera's view at the start pose, and the tool camera sees the part under the TCP. Real
    perception, real robot motion, closed loop: move a coloured object in front of the webcam.
    """

    ZOOM = 1.8

    def __init__(self, index, color):
        super().__init__(color)
        self.cam = Webcam(index)
        start = [*START_XY, FOLLOW_Z, *TOOL_DOWN]
        c = np.array([self.w / 2, self.h / 2])
        corners = [(0, 0), (self.w, 0), (self.w, self.h), (0, self.h)]
        self.table = np.array([[*self.pixel_to_table(*(c + self.ZOOM * (np.array(p) - c)), start), OBJECT_Z]
                               for p in corners])
        self.view = None

    def _webcam(self):
        frame, seq = self.cam.read(None, None)
        return cv2.flip(frame, 1), seq            # mirror, so it moves like a selfie view

    def read(self, tcp_pose, now):
        frame, _ = self._webcam()
        fh, fw = frame.shape[:2]
        src = np.array([(0, 0), (fw, 0), (fw, fh), (0, fh)], dtype=np.float32)
        dst = np.asarray(self.project(self.table, tcp_pose), dtype=np.float32).reshape(4, 2)
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(frame, M, (self.w, self.h), borderValue=(60, 60, 60))
        # footprint of the tool camera's view on the webcam frame, for the overview inset
        view = np.float32([(0, 0), (self.w, 0), (self.w, self.h), (0, self.h)]).reshape(-1, 1, 2)
        self.view = cv2.perspectiveTransform(view, np.linalg.inv(M)).reshape(-1, 2)
        self.seq += 1
        return img, self.seq

    def overview(self, width=200):
        """Small webcam image with the tool camera's footprint drawn on it."""
        frame, _ = self._webcam()
        frame = frame.copy()
        if self.view is not None:
            cv2.polylines(frame, [self.view.astype(np.int32)], True, (0, 255, 255), 3)
        return cv2.resize(frame, (width, int(width * frame.shape[0] / frame.shape[1])))

    def close(self):
        self.cam.close()


# ------------------------------------------------------------------ robot
def connect(ip):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    rtde_c = RTDEControlInterface(ip)
    rtde_r = RTDEReceiveInterface(ip)
    if rtde_r.getRobotMode() != 7 or rtde_r.getSafetyMode() != 1:
        raise RuntimeError(f'Robot not ready (mode {rtde_r.getRobotMode()}, safety {rtde_r.getSafetyMode()})')
    return rtde_c, rtde_r


def robot_fault(rtde_r):
    return rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped() or rtde_r.getSafetyMode() != 1


CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'follow_calibration.json')
HOME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'follow_home.json')
HOME_MAX_JOINT_MOVE_DEG = 45            # --home refuses a bigger joint move (no big sweeps)


def square_orientation(rotvec):
    """Snap a TCP orientation to the base axes: tool Z and X go to the nearest +-X/Y/Z, Y = Z x X."""
    R = cv2.Rodrigues(np.asarray(rotvec, float))[0]

    def snap(v, exclude=None):
        k = max((i for i in range(3) if i != exclude), key=lambda i: abs(v[i]))
        e = np.zeros(3); e[k] = np.sign(v[k])
        return e, k

    z, kz = snap(R[:, 2])
    x, _ = snap(R[:, 0], exclude=kz)
    Rs = np.column_stack([x, np.cross(z, x), z])
    return cv2.Rodrigues(Rs)[0].ravel(), math.degrees(math.acos(np.clip((np.trace(Rs.T @ R) - 1) / 2, -1, 1)))


def set_home(rtde_c, rtde_r, path):
    """Save the current TCP position with a squared-up orientation as the start pose. No motion."""
    import json
    pose, q_now = list(rtde_r.getActualTCPPose()), rtde_r.getActualQ()
    rv, angle = square_orientation(pose[3:])
    home = [*pose[:3], *rv]
    if not rtde_c.isPoseWithinSafetyLimits(home) or not rtde_c.getInverseKinematicsHasSolution(home, q_now):
        raise RuntimeError('the squared-up pose is not reachable from here')
    q = rtde_c.getInverseKinematics(home, q_now)
    move = np.degrees(np.abs(np.asarray(q) - q_now))
    R = pose_rotation(home)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dict(pose=home, q=list(q), created=time.strftime('%Y-%m-%d %H:%M:%S')), f, indent=2)
    axis = lambda v: ('+' if v[np.argmax(np.abs(v))] > 0 else '-') + 'XYZ'[int(np.argmax(np.abs(v)))]
    print(f'Home saved to {os.path.relpath(path)}: position {np.round(home[:3], 4).tolist()} m, '
          f'camera along {axis(R[:, 2])}, image x along {axis(R[:, 0])}, image y along {axis(R[:, 1])}')
    print(f'  orientation squared by {angle:.1f} deg; joints {np.round(np.degrees(q), 2).tolist()} '
          f'(largest change from now {move.max():.1f} deg). Use --home to start from it.')


def go_home(rtde_c, rtde_r, path):
    """Slow joint move to the saved home pose; refuses big moves."""
    import json
    with open(path) as f:
        home = json.load(f)
    q_now = np.asarray(rtde_r.getActualQ())
    move = np.degrees(np.abs(np.asarray(home['q']) - q_now))
    if move.max() > HOME_MAX_JOINT_MOVE_DEG:
        raise RuntimeError(f'home is {move.max():.0f} deg away on joint {int(np.argmax(move)) + 1} '
                           f'(limit {HOME_MAX_JOINT_MOVE_DEG}): jog the robot closer to it first')
    print(f'Moving (moveJ, slow) to home {np.round(home["pose"][:3], 3).tolist()} m '
          f'(largest joint change {move.max():.1f} deg)...')
    rtde_c.moveJ(home['q'], 0.25, 0.25)


def calibrate(rtde_c, rtde_r, cam, color, pose0, A=BASE_XY, step=0.05, shape=None):
    """Measure the image Jacobian J (px per m): move the TCP +-step along the two servo-plane axes
    (columns of A) and watch how a STILL object shifts in the image.
    Replaces CAM_YAW_DEG / CAM_FLIP / CAM_HFOV_DEG.

    Robust to a slightly moving object: each position is the median of 20 frames, and for each axis
    the object is measured at the start pose before and after the +-step moves. If it drifted, or the
    + and - shifts are not mirror images of each other, the object moved and calibration is refused.
    """
    if isinstance(cam, SimCamera):
        cam.paused, was_manual = True, cam.manual
        cam.manual = cam.object_xy()

    def measure(n=20, timeout=4.0):
        pts, last = [], None
        t_end = time.time() + timeout
        while len(pts) < n and time.time() < t_end:
            frame, seq = cam.read(rtde_r.getActualTCPPose(), time.time())
            if seq != last:
                last, d = seq, detect(frame, color, shape)
                if d is not None:
                    pts.append(d[:2])
            time.sleep(0.005)
        return np.median(pts, axis=0) if len(pts) >= 0.7 * n else None

    def go(pose):
        rtde_c.moveL(list(pose), MAX_SPEED, 0.2)
        time.sleep(0.4)          # let the camera's frames catch up with the robot

    def measure_or_fail(where):
        m = measure()
        if m is None:
            raise RuntimeError(f'object not seen {where}: keep it in the middle of the image')
        return m

    print(f'Calibrating: keep the object STILL. The TCP moves +-{step * 100:.0f} cm along '
          f'{np.round(A[:, 0], 2).tolist()} and {np.round(A[:, 1], 2).tolist()} (base frame)...')
    J = np.zeros((2, 2))
    try:
        for axis in (0, 1):
            m0a = measure_or_fail('at the start pose')
            seen = []
            for sign in (+1, -1):
                p = list(pose0)
                p[:3] = np.asarray(pose0[:3]) + sign * step * A[:, axis]
                go(p)
                seen.append(measure_or_fail(f'after moving {sign * step * 100:+.0f} cm along axis {axis + 1}'))
            go(pose0)
            m0b = measure_or_fail('back at the start pose')
            m0 = (m0a + m0b) / 2
            shift = np.linalg.norm(seen[0] - seen[1]) / 2          # px per step
            drift = np.linalg.norm(m0a - m0b)                      # object motion during this axis
            asym = np.linalg.norm((seen[0] - m0) + (seen[1] - m0))  # + and - shifts should cancel
            print(f'  axis {axis + 1}: shift {shift:.0f} px per {step * 100:.0f} cm, '
                  f'object drift {drift:.1f} px, asymmetry {asym:.1f} px')
            if shift < 10:
                raise RuntimeError(f'object barely moved in the image ({shift:.0f} px): too far away, '
                                   'or increase --calib-step')
            if drift > 0.25 * shift or asym > 0.3 * shift:
                raise RuntimeError('the object moved during calibration: rest it on something, '
                                   'or hold it really still, and try again')
            J[:, axis] = (seen[0] - seen[1]) / (2 * step)
    finally:
        go(pose0)
        if isinstance(cam, SimCamera):
            cam.paused, cam.manual = False, was_manual
    cols = np.linalg.norm(J, axis=0)
    if abs(np.linalg.det(J)) < 0.3 * cols.prod():
        raise RuntimeError(f'Calibration unreliable (J = {J.round(0).tolist()} px/m): '
                           'the two axes give almost the same image direction')
    describe_calibration(J, A, pose_rotation(pose0)[:, 2])
    distance_m = None
    if hasattr(cam, 'depth_at'):
        obj = measure()
        if obj is not None:
            distance_m = cam.depth_at(obj[0], obj[1], window=11)
            if distance_m is not None:
                print(f'  RealSense depth at object: {distance_m:.2f} m')
    return J, distance_m


def scene_shift(img0, img1, min_matches=20):
    """Median image displacement (px) of the scene between two frames, from ORB feature matches.

    Points that do not move (parts of the tool in view) are ignored; the median makes it robust to
    people moving in the scene. Returns None if there are too few matches.
    """
    orb = cv2.ORB_create(1500)
    g0, g1 = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (img0, img1))
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    if len(matches) < min_matches:
        return None
    d = np.array([np.subtract(k1[m.trainIdx].pt, k0[m.queryIdx].pt) for m in matches])
    med = np.median(d, axis=0)
    if np.linalg.norm(med) > 1.0:          # ignore what moves with the camera (stays at 0 px)
        d = d[np.linalg.norm(d, axis=1) > 0.5]
    # keep the matches that agree with the dominant motion, then take their median
    med = np.median(d, axis=0)
    good = d[np.linalg.norm(d - med, axis=1) < max(3.0, 0.3 * np.linalg.norm(med))]
    return np.median(good, axis=0) if len(good) >= min_matches // 2 else None


def calibrate_scene(rtde_c, rtde_r, cam, pose0, A, K, distance, step=0.05):
    """Like calibrate(), but from the motion of the whole scene: no object needed.

    The scene's shift gives the image directions (camera rotation, mirroring, signs) exactly. Its
    magnitude depends on how far the background is, so the scale is set from the focal length and
    the expected object distance instead: J = direction * f / distance.
    """
    def grab():
        time.sleep(0.5)                    # let the camera's frames catch up with the robot
        return cam.read(rtde_r.getActualTCPPose(), time.time())[0].copy()

    def go(pose):
        rtde_c.moveL(list(pose), MAX_SPEED, 0.2)

    print(f'Calibrating from the scene (no object needed). The TCP moves +-{step * 100:.0f} cm along '
          f'{np.round(A[:, 0], 2).tolist()} and {np.round(A[:, 1], 2).tolist()} (base frame)...')
    J_scene = np.zeros((2, 2))
    try:
        for axis in (0, 1):
            go(pose0); f0 = grab()
            shifts = []
            for sign in (+1, -1):
                p = list(pose0)
                p[:3] = np.asarray(pose0[:3]) + sign * step * A[:, axis]
                go(p)
                d = scene_shift(f0, grab())
                if d is None:
                    raise RuntimeError('not enough texture in the scene to measure its motion')
                shifts.append(d)
            go(pose0)
            back = scene_shift(f0, grab())
            shift = np.linalg.norm(shifts[0] - shifts[1]) / 2
            asym = np.linalg.norm(shifts[0] + shifts[1])
            drift = np.linalg.norm(back) if back is not None else float('inf')
            print(f'  axis {axis + 1}: scene shift {shift:.1f} px per {step * 100:.0f} cm, '
                  f'back-at-start {drift:.1f} px, asymmetry {asym:.1f} px')
            if shift < 3:
                raise RuntimeError(f'the scene barely moved ({shift:.1f} px): increase --calib-step')
            if drift > 0.25 * shift or asym > 0.3 * shift:
                raise RuntimeError('inconsistent scene motion (things moving in view?): try again')
            J_scene[:, axis] = (shifts[0] - shifts[1]) / (2 * step)
    finally:
        go(pose0)
    cols = np.linalg.norm(J_scene, axis=0)
    if abs(np.linalg.det(J_scene)) < 0.3 * cols.prod():
        raise RuntimeError(f'unreliable result (J = {J_scene.round(0).tolist()} px/m)')
    J = J_scene / cols.mean() * K[0, 0] / distance
    print(f'  background at ~{K[0, 0] / cols.mean():.1f} m; scale set for an object at {distance:.2f} m')
    describe_calibration(J, A, pose_rotation(pose0)[:, 2])
    return J, distance


def measure_object_distance(cam, fallback, frame=None):
    """Depth at the image centre if the camera supports it; else `fallback` (--object-distance)."""
    if not hasattr(cam, 'depth_at'):
        return fallback
    if frame is None:
        frame, _ = cam.read(None, 0.0)
    h, w = frame.shape[:2]
    d = cam.depth_at(w / 2, h / 2, window=31)
    if d is None:
        print(f'  No valid RealSense depth at image centre; using --object-distance {fallback:.2f} m')
        return fallback
    print(f'  RealSense depth at image centre: {d:.2f} m (used for scale)')
    return d


def reachable_box(rtde_c, start, plane, radius, q_now, min_z=0.10, step=0.02, margin=0.02, max_joint_change_deg=90):
    """Shrink the +-radius box (plane coordinates) to what the arm can reach from the start pose.

    Walks outward from the start along each plane direction in `step` increments and stops at the
    first pose that is below min_z (base frame: the table), outside the safety limits, has no IK
    solution, or needs a joint to turn more
    than max_joint_change_deg from now (a flip / near a singularity). Then checks the corners
    (4 in 2D, 8 in 3D) and pulls in any that fail. Uses getInverseKinematicsHasSolution first: asking
    getInverseKinematics for an unreachable pose stops the robot's control script.
    Returns (lo, hi): per-axis limits, in the coordinates of the columns of `plane`.
    """
    origin, q_now = np.asarray(start[:3], float), np.asarray(q_now, float)

    def ok(c):
        pose = [*(origin + plane @ np.asarray(c, float)), *start[3:]]
        if pose[2] < min_z:
            return False
        if not rtde_c.isPoseWithinSafetyLimits(pose):
            return False
        if not rtde_c.getInverseKinematicsHasSolution(pose, list(q_now)):
            return False
        q = rtde_c.getInverseKinematics(pose, list(q_now))
        return bool(q) and np.max(np.abs(np.asarray(q) - q_now)) < math.radians(max_joint_change_deg)

    import itertools
    n = plane.shape[1]
    lo, hi = np.full(n, -radius), np.full(n, radius)
    for axis in range(n):
        for sign, lim in ((+1, hi), (-1, lo)):
            reach = 0.0
            for t in [*np.arange(step, radius - 1e-9, step), radius]:
                c = np.zeros(n); c[axis] = sign * t
                if not ok(c):
                    break
                reach = t
            lim[axis] = sign * (max(reach - margin, 0.0) if reach < radius else radius)
    for _ in range(10):                           # corners: shrink the failing ones
        bad = [sel for sel in itertools.product((0, 1), repeat=n)
               if not ok([(lo, hi)[k][a] for a, k in enumerate(sel)])]
        if not bad:
            break
        for sel in bad:
            for a, k in enumerate(sel):
                (lo, hi)[k][a] *= 0.85
    return lo, hi


def describe_calibration(J, A, view):
    """Print J as scale + rotation + mirror, to compare with the mounting you expect."""
    cols = np.linalg.norm(J, axis=0)
    # an unmirrored camera gives det(J) the sign of (a1 x a2) . view direction
    handed = np.sign(np.cross(A[:, 0], A[:, 1]) @ view)
    mirrored = np.sign(np.linalg.det(J)) != handed
    print(f'Image Jacobian (px per m): {J.round(0).tolist()}')
    print(f'  ~{1000 / cols.mean():.2f} mm per pixel at the object\'s distance, '
          f'image x axis = plane ({J[0, 0] / cols[0]:+.2f}, {J[0, 1] / cols[1]:+.2f}) direction, '
          f'{"mirrored" if mirrored else "not mirrored"}')


def save_calibration(path, J, A, pose0, step, distance_m=None):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = dict(J=J.tolist(), plane=np.asarray(A).tolist(), view=pose_rotation(pose0)[:, 2].tolist(),
                pose=list(pose0), step=step, created=time.strftime('%Y-%m-%d %H:%M:%S'))
    if distance_m is not None:
        data['distance_m'] = float(distance_m)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Calibration saved to {os.path.relpath(path)}'
          + (f' (distance_m={distance_m:.2f})' if distance_m is not None else ''))


def load_calibration(path, A, pose0, max_angle_deg=30):
    """Saved J, expressed in the current servo plane.

    The camera is fixed to the tool, so the calibration is re-expressed in TOOL axes: that part does
    not change when the tool is turned (e.g. rolled 180 deg to put the image the right way up).
    Only the scale depends on the object's distance. A warning is printed if the camera now looks
    in a very different direction, since the distance to what it sees probably changed too.
    Returns (J, distance_m); distance_m may be None for older calib files.
    """
    import json
    with open(path) as f:
        data = json.load(f)
    R_cal = pose_rotation(data['pose'])
    # pixel shift per metre of motion along tool X / tool Y at calibration time
    M_tool = np.asarray(data['J']) @ (np.asarray(data['plane']).T @ R_cal[:, :2])
    R_now = pose_rotation(pose0)
    J = M_tool @ (R_now[:, :2].T @ np.asarray(A))
    view_now = R_now[:, 2]
    angle = math.degrees(math.acos(np.clip(np.dot(view_now, data['view']), -1, 1)))
    distance_m = data.get('distance_m')
    print(f'Loaded calibration from {os.path.relpath(path)} ({data["created"]}); camera direction '
          f'{angle:.1f} deg from calibration time'
          + (f'; distance_m={distance_m:.2f}' if distance_m is not None else ''))
    if angle > max_angle_deg:
        print(f'  Warning: the camera looks {angle:.0f} deg away from where it was calibrated: the object '
              'distance (scale) may differ. If the robot reacts too strongly or too weakly, recalibrate.')
    describe_calibration(J, A, view_now)
    return J, distance_m


def home_velocity(target, v_prev, dt):
    """--hand going home: velocity that brings `target` (offset from the start pose along the motion
    axes) back to zero, at most MAX_SPEED, slowing down near it, with the MAX_ACCEL limit."""
    target = np.asarray(target, float)
    dist = float(np.linalg.norm(target))
    v_des = np.zeros_like(target) if dist < 1e-3 else -target / dist * min(MAX_SPEED, HOME_GAIN * dist)
    dv = v_des - np.asarray(v_prev, float)
    max_dv = MAX_ACCEL * dt
    if np.linalg.norm(dv) > max_dv:
        dv *= max_dv / np.linalg.norm(dv)
    return np.asarray(v_prev, float) + dv


# ------------------------------------------------------------------ main loop
def draw_overlay(img, det, K, follower, v, status):
    h, w = img.shape[:2]
    cx, cy = int(K[0, 2]), int(K[1, 2])
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    if det is not None:
        p = (int(det[0]), int(det[1]))
        if det[3] is not None:
            cv2.drawContours(img, [det[3]], -1, (0, 255, 0), 2)
        cv2.circle(img, p, 8, (255, 255, 255), 2)
        cv2.line(img, (cx, cy), p, (255, 255, 255), 1)
    depth = f'  d={follower.depth_now:.2f}m' if follower.depth_now is not None else ''
    cv2.putText(img, f'{status}  v=({v[0]*100:+.1f}, {v[1]*100:+.1f}) cm/s{depth}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    global GAIN, MAX_SPEED, FOLLOW_Z, FEEDFORWARD
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sim', action='store_true', help='synthetic tool camera (closed loop with URSim)')
    ap.add_argument('--webcam-table', action='store_true',
                    help='PC webcam image laid on the virtual table, seen by the tool camera (closed loop with URSim)')
    ap.add_argument('--no-robot', action='store_true', help='do not connect to the robot')
    ap.add_argument('--camera', default='0',
                    help="index, device path, or 'realsense' (librealsense colour+depth)")
    ap.add_argument('--color', default='red', choices=list(COLOR_RANGES))
    ap.add_argument('--shape', nargs='?', const='auto', default=None, metavar='PHOTO',
                    help='also match the silhouette from PHOTO (default: newest image in data/). '
                         'The object must be the largest blob of --color in that photo')
    ap.add_argument('--hand', action='store_true',
                    help='follow a hand (MediaPipe, open or closed) instead of a colour; no hand sends '
                         'the robot back to where following started')
    ap.add_argument('--hand-model', default=HAND_MODEL, help='MediaPipe gesture_recognizer.task')
    ap.add_argument('--gripper', choices=['none', 'robotiq', 'tool-do'], default='none',
                    help='with --hand: a fist closes the claw, an open hand opens it. robotiq = Robotiq '
                         '2F-85/Hand-E through its URCap (port 63352); tool-do = tool digital output 0 '
                         '(the simulated gripper, works in URSim)')
    ap.add_argument('--ip', default=ROBOT_IP)
    ap.add_argument('--gain', type=float, default=GAIN)
    ap.add_argument('--feedforward', type=float, default=FEEDFORWARD)
    ap.add_argument('--max-speed', type=float, default=MAX_SPEED)
    ap.add_argument('--height', type=float, default=FOLLOW_Z)
    ap.add_argument('--start-here', action='store_true',
                    help='real robot: follow from the CURRENT pose (height + orientation), no move to a preset pose')
    ap.add_argument('--set-home', action='store_true',
                    help='save the current position, with the orientation squared to the base axes, as the '
                         'home pose (no motion), then exit')
    ap.add_argument('--home', action='store_true',
                    help='move slowly to the saved home pose, then follow from there (implies --start-here)')
    ap.add_argument('--home-file', default=HOME_FILE, help='where --set-home saves / --home reads')
    ap.add_argument('--radius', type=float, default=0.25,
                    help='with --start-here: how far (m) the target may go from the start along each '
                         'plane axis; reduced automatically where the arm cannot reach')
    ap.add_argument('--follow-depth', action='store_true',
                    help='also move along the camera axis to keep the object at --target-distance '
                         '(needs a depth camera, e.g. --camera realsense, and --start-here or --home)')
    ap.add_argument('--target-distance', type=float, default=0.5,
                    help='m, camera-object distance kept with --follow-depth')
    ap.add_argument('--depth-gain', type=float, default=1.0,
                    help='1/s, gain of the distance controller (velocity = gain x distance error)')
    ap.add_argument('--min-z', type=float, default=0.10,
                    help='with --start-here: lowest TCP height (m, robot base frame) the target may reach, '
                         'to stay clear of the table the robot stands on')
    ap.add_argument('--calibrate', action='store_true',
                    help='measure the camera->robot mapping first (object must be visible and still)')
    ap.add_argument('--calibrate-scene', action='store_true',
                    help='calibrate from the motion of the whole scene: no object needed (hold it or not)')
    ap.add_argument('--object-distance', type=float, default=0.5,
                    help='m, fallback camera-object distance for --calibrate-scene when depth is unavailable')
    ap.add_argument('--calib-step', type=float, default=0.05,
                    help='m, size of the calibration moves (bigger = less sensitive to a moving object)')
    ap.add_argument('--load-calib', action='store_true',
                    help='use the calibration saved by the last --calibrate (no calibration moves)')
    ap.add_argument('--calib-file', default=CALIB_FILE, help='where --calibrate saves / --load-calib reads')
    ap.add_argument('--sim-mount', type=float, nargs=2, metavar=('YAW', 'FLIP'),
                    help='--sim only: true camera mounting (deg, 0/1), to test --calibrate')
    ap.add_argument('--headless', action='store_true', help='no window (for tests)')
    ap.add_argument('--duration', type=float, default=0, help='stop after N seconds (0 = run until q)')
    args = ap.parse_args()
    GAIN, MAX_SPEED, FOLLOW_Z, FEEDFORWARD = args.gain, min(args.max_speed, 0.08), args.height, args.feedforward
    if (args.sim or args.webcam_table) and args.no_robot:
        sys.exit('--sim / --webcam-table need the robot pose; drop --no-robot')

    if args.hand:
        if args.sim:
            sys.exit('--hand needs a real camera: --sim only renders a coloured disc (try --webcam-table)')
        if args.calibrate:
            sys.exit('--calibrate measures a coloured object; with --hand use --calibrate-scene or --load-calib')
        if args.shape:
            sys.exit('--shape is for colour detection; drop it with --hand')
        try:
            hand_det = AsyncHandDetector(HandDetector(args.hand_model))
        except (OSError, RuntimeError) as e:
            sys.exit(f'--hand: {e}')
        gate = HandGate()
        print('Hand mode: show a hand (open or closed) to be followed; hide it to send the robot home.')
    else:
        hand_det = gate = None
    claw = None
    if args.gripper != 'none':
        if not args.hand:
            sys.exit('--gripper copies the hand (fist = close): it needs --hand')
        if args.no_robot:
            sys.exit('--gripper needs the robot: drop --no-robot')
        try:
            claw = AsyncGripper(RobotiqGripper(args.ip) if args.gripper == 'robotiq' else ToolDOGripper(args.ip))
        except (OSError, RuntimeError) as e:
            hand_det.close()
            sys.exit(f'--gripper {args.gripper}: {e}')
        print(f'Gripper ({args.gripper}): a fist held {GRIP_CONFIRM} s closes it, an open hand opens it.')

    if args.home:
        args.start_here = True             # follow from home exactly like from the current pose
    local = args.ip in ('127.0.0.1', 'localhost')
    if not local and not args.no_robot and not (args.start_here or args.set_home):
        sys.exit('Real robot: use --start-here (jog the camera above the table first). '
                 'The preset start pose and workspace are for URSim only.')

    shape = None
    if args.shape:
        try:
            shape = learn_shape(resolve_shape_photo(args.shape), args.color)
        except (OSError, RuntimeError) as e:
            sys.exit(f'--shape: {e}')

    rtde_c = rtde_r = None
    start = [*START_XY, FOLLOW_Z, *TOOL_DOWN]
    if not args.no_robot:
        rtde_c, rtde_r = connect(args.ip)
        if args.set_home or args.home:
            try:
                (set_home if args.set_home else go_home)(rtde_c, rtde_r, args.home_file)
            except (OSError, RuntimeError) as e:
                print(f'{"--set-home" if args.set_home else "--home"} failed: {e}')
                rtde_c.stopScript(); sys.exit(1)
            if args.set_home:
                rtde_c.stopScript(); rtde_c.disconnect(); rtde_r.disconnect()
                return
        if args.start_here:
            start = list(rtde_r.getActualTCPPose())
            R0 = pose_rotation(start)
            plane = R0[:, :2]                  # tool X and Y: at right angles to the camera's view
            looking_down = R0[2, 2] < -0.9
            if not looking_down and not (args.calibrate or args.calibrate_scene or args.load_calib):
                rtde_c.stopScript()
                sys.exit('The camera does not look down: the default mapping assumes it does. '
                         'Add --calibrate (object still), --calibrate-scene (no object), or --load-calib.')
            print(f'Following from the current pose {np.round(start[:3], 3).tolist()} m, '
                  f'camera looking {"down" if looking_down else "along " + str(np.round(R0[:, 2], 2).tolist())}; '
                  f'moving in the plane of {np.round(plane[:, 0], 2).tolist()} and '
                  f'{np.round(plane[:, 1], 2).tolist()}, +-{args.radius:.2f} m')
        else:
            seed = [math.radians(a) for a in HOME_SEED_DEG]
            q = rtde_c.getInverseKinematics(start, seed)
            if not q or max(abs(a - b) for a, b in zip(q, seed)) > math.radians(60):
                raise RuntimeError(f'No elbow-up IK solution near HOME_SEED_DEG for {start}')
            print(f'Moving (moveJ, slow) to start pose {START_XY} at z={FOLLOW_Z} m...')
            rtde_c.moveJ(q, 0.25, 0.25)  # rad/s, rad/s² — slow approach for safety
    if not args.start_here:
        plane = BASE_XY
    origin = np.asarray(start[:3], float)
    if args.follow_depth and not args.start_here:
        if rtde_c:
            rtde_c.stopScript()
        sys.exit('--follow-depth needs --start-here or --home (it moves along the camera axis).')
    if args.follow_depth and not 0.2 <= args.target_distance <= DEPTH_MAX_M:
        sys.exit(f'--target-distance must be between 0.2 and {DEPTH_MAX_M} m')
    # motion frame: the two plane axes, plus the camera's view axis (tool Z) with --follow-depth
    motion = np.column_stack([plane, pose_rotation(start)[:, 2]]) if args.follow_depth else plane

    box_lo, box_hi = np.full(motion.shape[1], -args.radius), np.full(motion.shape[1], args.radius)
    if args.start_here and rtde_c:
        if start[2] < args.min_z:
            rtde_c.stopScript()
            sys.exit(f'The TCP is at z={start[2]:.3f} m, below --min-z {args.min_z} m: jog it higher first.')
        box_lo, box_hi = reachable_box(rtde_c, start, motion, args.radius, rtde_r.getActualQ(), min_z=args.min_z)
        names = ['axis 1', 'axis 2', 'camera axis (+ = toward the object)']
        print('Allowed region around the start (m): '
              + ', '.join(f'{names[k]} [{box_lo[k]:+.2f}, {box_hi[k]:+.2f}]' for k in range(len(box_lo)))
              + ('' if np.allclose(box_hi - box_lo, 2 * args.radius) else
                 f'  (reduced from +-{args.radius:.2f}: out of reach, or below --min-z {args.min_z} m)'))

    def clamp_plane(c):
        """Keep the target inside the allowed box (--start-here), or WORKSPACE (base XY)."""
        if args.start_here:
            return np.clip(c, box_lo, box_hi)
        c = c[:2]
        x, y = clamp_xy(*(origin[:2] + c))
        return np.array([x, y]) - origin[:2]

    if args.webcam_table:
        cam = WebcamTable(args.camera, args.color)
    elif args.sim:
        cam = SimCamera(args.color, mount=args.sim_mount)
    else:
        print(f'Opening camera {args.camera!r}...', flush=True)
        try:
            cam = open_camera(args.camera)
        except RuntimeError as e:
            print(f'Camera failed: {e}', flush=True)
            if rtde_c:
                rtde_c.stopScript()
                rtde_c.disconnect()
            if rtde_r:
                rtde_r.disconnect()
            sys.exit(1)
    window = 'Follow object (q to quit)'
    state = {'pose': start}
    if not args.headless:
        cv2.namedWindow(window)
        if args.sim:
            def on_mouse(event, x, y, flags, _):
                if event == cv2.EVENT_LBUTTONDOWN:
                    cam.manual = cam.pixel_to_table(x, y, state['pose'])
            cv2.setMouseCallback(window, on_mouse)

    follower, K = Follower(plane, origin), None
    if args.calibrate:
        if not rtde_c:
            sys.exit('--calibrate needs the robot')
        try:
            J, distance_m = calibrate(rtde_c, rtde_r, cam, args.color, start, plane,
                                      step=args.calib_step, shape=shape)
        except RuntimeError as e:
            print(f'Calibration failed: {e}')
            rtde_c.stopScript(); cam.close()
            sys.exit(1)
        follower.set_jacobian(J, distance_m)
        save_calibration(args.calib_file, follower.J_calib, plane, start, args.calib_step,
                         distance_m=follower.distance_m)
    elif args.calibrate_scene:
        if not rtde_c:
            sys.exit('--calibrate-scene needs the robot')
        frame0 = cam.read(state['pose'], 0.0)[0]
        distance = measure_object_distance(cam, args.object_distance, frame=frame0)
        try:
            J, distance_m = calibrate_scene(
                rtde_c, rtde_r, cam, start, plane, camera_matrix(cam, frame0),
                distance, step=args.calib_step)
        except RuntimeError as e:
            print(f'Calibration failed: {e}')
            rtde_c.stopScript(); cam.close()
            sys.exit(1)
        follower.set_jacobian(J, distance_m)
        save_calibration(args.calib_file, follower.J_calib, plane, start, args.calib_step,
                         distance_m=follower.distance_m)
    elif args.load_calib:
        try:
            J, distance_m = load_calibration(args.calib_file, plane, start)
        except (OSError, RuntimeError) as e:
            print(f'Cannot use the saved calibration: {e}')
            if rtde_c:
                rtde_c.stopScript()
            cam.close()
            sys.exit(1)
        follower.set_jacobian(J, distance_m)
    if args.follow_depth and not hasattr(cam, 'depth_at'):
        if rtde_c:
            rtde_c.stopScript()
        cam.close()
        sys.exit('--follow-depth needs a camera with depth (use --camera realsense)')
    depth_follower = DepthFollower(args.target_distance, args.depth_gain) if args.follow_depth else None
    target = np.zeros(motion.shape[1])   # position along the motion axes, relative to the start pose
    last_seq, det, frame = 0, None, None
    notice = None                        # (text, until): short message drawn in the window
    hand, hand_count, hand_time, v_prev = None, 0, 0.0, np.zeros(motion.shape[1])
    errors, t0, last_print, tick = [], time.time(), 0.0, 0
    # Pace the loop to DT with our own clock. Not rtde_c.initPeriod()/waitPeriod(): those wait for the
    # RTDE interface's period (2 ms at the e-series' 500 Hz), so the loop ran at ~130 Hz and the
    # target, advanced by v * DT per step, moved 2-3x faster than --max-speed.
    next_tick, last_step = time.perf_counter(), None
    try:
        while True:
            step_start = time.perf_counter()
            dt = DT if last_step is None else min(step_start - last_step, 2 * DT)   # measured step
            last_step = step_start
            now = time.time() - t0
            pose = rtde_r.getActualTCPPose() if rtde_r else state['pose']
            state['pose'] = pose
            if rtde_r and robot_fault(rtde_r):
                print('Robot fault (protective / emergency stop): stopping'); break

            frame, seq = cam.read(pose, now)
            if K is None:
                K = camera_matrix(cam, frame)
            new = seq != last_seq
            d = None
            if hand_det:   # detection runs in its own thread: "new" = a new hand result, not a new frame
                if new:
                    hand_det.submit(frame)
                    last_seq = seq
                count, latest = hand_det.latest()
                new = count != hand_count
                if new:    # any visible hand is the object; no hand is "nothing to follow"
                    hand_count, hand, hand_time = count, latest, now
                    mode = gate.update(hand.state if hand else None, now)
                    det = hand.as_detection() if hand is not None and mode == 'follow' else None
                    if claw and gate.grip is not None:   # sent from its own thread, only on a change
                        claw.request(gate.grip == 'closed')
                elif now - hand_time > LOST_TIMEOUT:   # detector stalled: same as no hand
                    hand = None
                    gate.update(None, now)
            elif new:
                det, last_seq = detect(frame, args.color, shape), seq
            if new:
                if det is not None and hasattr(cam, 'depth_at'):
                    d = cam.depth_at(det[0], det[1], window=11)
                    if follower.J_calib is not None:
                        follower.apply_depth(d)
                    else:
                        follower.depth_now = d
            v = follower.update(det if new else None, K, pose, now)
            if depth_follower:
                # Centring the object in the image gets the speed budget first (losing it from view
                # loses everything); the camera axis only gets what is left of --max-speed.
                vz = depth_follower.update(d if new else None, now)
                budget = math.sqrt(max(MAX_SPEED ** 2 - float(np.dot(v, v)), 0.0))
                v = np.append(v, np.clip(vz, -budget, budget))

            if gate and gate.mode == 'home':
                # back to the start pose along the motion axes; the controllers take over this velocity
                # so following resumes smoothly when the hand opens again
                v = home_velocity(target, v_prev, dt)
                follower.v = v[:2].copy()
                if depth_follower:
                    depth_follower.v = float(v[2])

            proposed = target + v * dt           # measured dt: the speed cap holds even if the loop is late
            target = clamp_plane(proposed)
            # Anti-windup: an axis pressed against its box edge stops pushing, and its controller
            # forgets the velocity it had built up (it would otherwise keep the speed budget).
            pinned = np.abs(target - proposed[:len(target)]) > 1e-9
            if pinned.any():
                v = v.copy(); v[:len(target)][pinned] = 0.0
                for k in np.flatnonzero(pinned):
                    if k < 2:
                        follower.v[k] = 0.0
                    elif depth_follower:
                        depth_follower.v = 0.0
            v_prev = v
            if rtde_c:   # orientation stays the start orientation
                rtde_c.servoL([*(origin + motion @ target), *start[3:]], 0.0, 0.0, DT, 0.1, 300)

            if args.sim:  # ground-truth error: object vs. point under the camera
                below = cam.pixel_to_table(K[0, 2], K[1, 2], pose)
                errors.append((now, np.linalg.norm(below - cam.object_xy())))

            status = 'TRACKING' if follower.tracking else 'LOST'
            if gate:
                status = ('FOLLOW' if gate.mode == 'follow' else 'HOME') + f'  hand {hand.state if hand else "none"}'
                if claw and claw.state is not None:
                    status += f'  claw {"closed" if claw.state else "open"}'
            if det is not None and det[4] is not None:
                status += f'  shape {det[4]:.2f}'
            if now - last_print > 1.0:
                last_print = now
                extra = f'  err {errors[-1][1]*1000:5.1f} mm' if errors else ''
                depth = f'  d={follower.depth_now:.2f}m' if follower.depth_now is not None else ''
                if depth_follower:
                    dd = depth_follower.d
                    depth = (f'  d={dd:.3f}m (target {args.target_distance:.2f})' if dd is not None
                             else '  d=?  (no depth: not moving along the camera axis)')
                vs = ', '.join(f'{x:+.3f}' for x in v)
                print(f't={now:5.1f}s {status:8s} tcp=({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}) '
                      f'v=({vs}) m/s{depth}{extra}')

            tick += 1
            if not args.headless and tick % 2 == 0:   # display at 25 Hz
                shown = draw_overlay(frame.copy(), det, K, follower, v, status)
                if hand is not None:
                    draw_hand(shown, hand, gate.mode)
                if notice and now < notice[1]:
                    cv2.putText(shown, notice[0], (10, shown.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2, cv2.LINE_AA)
                if args.webcam_table:   # inset: webcam with the tool camera's footprint
                    inset = cam.overview()
                    shown[-inset.shape[0] - 10:-10, -inset.shape[1] - 10:-10] = inset
                cv2.imshow(window, shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(' ') and args.sim:
                    cam.paused = not cam.paused
                    cam.manual = None
                if key == ord('s') and not hand_det:          # learn the shape from what the camera sees now
                    try:
                        shape, path = capture_shape(frame, args.color)
                        print(f'Shape photo saved to {path} (--shape with no path reuses it)')
                        notice = ('Shape learned: following only this silhouette', now + 2.5)
                    except RuntimeError as e:
                        print(f'Shape capture failed: {e}')
                        notice = (f'Shape capture failed: no {args.color} object', now + 2.5)
                if key == ord('x') and shape is not None:
                    shape = None
                    print('Shape forgotten: following by colour only')
                    notice = ('Shape forgotten: colour only', now + 2.5)
            if args.duration and now > args.duration:
                break
            next_tick += DT
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()     # late: don't try to catch up with a burst
    except KeyboardInterrupt:
        pass
    finally:
        if rtde_c:
            rtde_c.servoStop(2.0)
            rtde_c.stopScript()
            rtde_c.disconnect()
        if rtde_r:
            rtde_r.disconnect()
        cam.close()
        if hand_det:
            hand_det.close()
        if claw:
            claw.close()
        cv2.destroyAllWindows()

    if errors:
        settled = [e for t, e in errors if t > 5.0] or [e for _, e in errors]
        print(f'Tracking error after 5 s: mean {np.mean(settled)*1000:.1f} mm, '
              f'max {np.max(settled)*1000:.1f} mm ({len(settled)} samples)')


if __name__ == '__main__':
    main()
