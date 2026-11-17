"""Make the UR5 follow a coloured object with an eye-in-hand camera (image-based visual servoing).

The camera is mounted on the tool and looks along the tool Z axis (down at the table). Each frame:
  1. detect the object by colour (HSV, same pipeline as Day 2),
  2. measure its offset from the image centre,
  3. turn that offset into a TCP velocity in the robot base frame (proportional control),
  4. integrate the velocity into a target pose and stream it with servoL at 50 Hz.
The TCP keeps a fixed height and orientation; only X and Y are servoed.

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
  --calibrate      Before following, move +-2 cm in X and Y and measure how a still object moves in
                   the image (image Jacobian). Use it on the real robot: no need to know how the
                   camera is mounted.
  --no-robot       Detection and computed velocity only, no robot connection (tune HSV ranges here).

Examples (URSim running, see ursim/bringup.sh):
    uv run python project/follow_object.py --sim
    uv run python project/follow_object.py --sim --headless --duration 30   # self-test, prints error
    uv run python project/follow_object.py --webcam-table --color red   # PC webcam + URSim
    uv run python project/follow_object.py --no-robot --color blue      # webcam, no robot
    uv run python project/follow_object.py --no-robot --color blue --camera realsense
See README.md (repository root) for every option and parameter.
Real robot (camera on the tool, object visible and still, hand on the e-stop):
    uv run python project/follow_object.py --ip <ROBOT_IP> --camera realsense --start-here --calibrate \
        --max-speed 0.05 --gain 1.5 --color red
Keys: q / Esc quit, space pause (sim object motion), click: move the sim object.
"""
import argparse
import math
import os
import sys
import threading
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')   # OpenCV's Qt only ships the X11 plugin
import cv2
import numpy as np

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
GAIN = 3.0                              # 1/s, proportional gain: v = GAIN * offset (+ feedforward)
FEEDFORWARD = 1.0                       # 0..1, add the estimated object velocity (0 = pure P control)
VEL_FILTER = 0.15                       # 0..1, low-pass on the object velocity estimate
MAX_SPEED = 0.15                        # m/s, TCP speed cap
MAX_ACCEL = 0.5                         # m/s², limit on velocity changes
DEADBAND_PX = 4                         # px, ignore smaller errors (detection jitter)
ERROR_FILTER = 0.5                      # 0..1, low-pass on the measured offset (1 = no filter)
LOST_TIMEOUT = 0.5                      # s without detection before the robot stops

CAM_HFOV_DEG = 60.0                     # horizontal field of view, used for the focal length
CAM_YAW_DEG = 0.0                       # camera rotation about the tool Z axis (mounting)
CAM_FLIP = False                        # True if the image is mirrored relative to the tool frame

# HSV ranges (OpenCV: H 0..179); red wraps around so it needs two ranges
COLOR_RANGES = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
    'green':  [((40, 80, 50), (85, 255, 255))],
    'blue':   [((100, 150, 50), (130, 255, 255))],
}
DRAW_BGR = {'red': (0, 0, 255), 'yellow': (0, 220, 220), 'green': (0, 200, 0), 'blue': (255, 0, 0)}
MIN_AREA = 300                          # px²


# ------------------------------------------------------------------ perception
def detect(img, color):
    """Largest blob of `color`: (cx, cy, area) in pixels, or None."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea, default=None)
    if best is None or cv2.contourArea(best) < MIN_AREA:
        return None
    m = cv2.moments(best)
    return m['m10'] / m['m00'], m['m01'] / m['m00'], m['m00']


def intrinsics(w, h, hfov_deg=CAM_HFOV_DEG):
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


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
class Follower:
    """Image offset -> base-frame XY velocity, with deadband, filter, speed and accel limits."""

    def __init__(self):
        self.err = None          # filtered offset (normalised image coordinates)
        self.v = np.zeros(2)     # current commanded XY velocity, m/s
        self.last_seen = -1e9
        self.obj = None          # (time, estimated object XY in the base frame)
        self.v_obj = np.zeros(2) # filtered object velocity estimate, m/s
        self.J = None            # measured image Jacobian (px per m of TCP motion), see calibrate()
        self.f = None            # focal length, px

    def offset_base(self, err, tcp_pose):
        """Normalised image offset -> XY offset (m) from the point under the camera to the object."""
        if self.J is not None:
            # moving the TCP by d shifts the object by J d pixels: it is centred when J d = -offset
            return -np.linalg.solve(self.J, np.asarray(err) * self.f)
        depth = max(tcp_pose[2] - OBJECT_Z, 0.05)
        offset_cam = np.array([err[0] * depth, err[1] * depth, 0.0])
        return (pose_rotation(tcp_pose) @ tool_to_cam() @ offset_cam)[:2]

    def update(self, det, K, tcp_pose, now):
        self.f = K[0, 0]
        if det is not None:
            u, v = det[0] - K[0, 2], det[1] - K[1, 2]
            e = np.array([u / K[0, 0], v / K[1, 1]])
            self.err = e if self.err is None else ERROR_FILTER * e + (1 - ERROR_FILTER) * self.err
            # object position = point under the camera + offset; its derivative is the object velocity
            obj = np.asarray(tcp_pose[:2]) + self.offset_base(e, tcp_pose)
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
            err = self.err if np.linalg.norm(self.err) * K[0, 0] >= DEADBAND_PX else np.zeros(2)
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


def clamp_xy(x, y):
    return (min(max(x, WORKSPACE['x'][0]), WORKSPACE['x'][1]),
            min(max(y, WORKSPACE['y'][0]), WORKSPACE['y'][1]))


# ------------------------------------------------------------------ cameras
def resolve_camera(name):
    """'0' -> 0, 'realsense' -> the RealSense colour stream, anything else -> device path."""
    name = str(name)
    if name.isdigit():
        return int(name)
    if name.lower() == 'realsense':
        # by-id names include the serial number only when the camera reports it, so match loosely;
        # video-index0 is the colour stream (index2 is infrared)
        import glob
        paths = sorted(glob.glob('/dev/v4l/by-id/*RealSense*-video-index0'))
        if not paths:
            raise RuntimeError('No RealSense found under /dev/v4l/by-id (check the USB cable)')
        return paths[0]
    return name


class Webcam:
    """Grabs frames in a thread so the 50 Hz control loop never blocks on the camera."""

    def __init__(self, index):
        index = resolve_camera(index)
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera {index}')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError('Camera read failed')
        self.frame, self.seq, self.running = frame, 1, True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.seq = frame, self.seq + 1

    def read(self, tcp_pose, now):
        with self.lock:
            return self.frame, self.seq

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


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

    def read(self, tcp_pose, now):
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
        src = np.float32([(0, 0), (fw, 0), (fw, fh), (0, fh)])
        dst = np.float32(self.project(self.table, tcp_pose))
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


def calibrate(rtde_c, rtde_r, cam, color, pose0, step=0.02):
    """Measure the image Jacobian J (px per m): move the TCP +-step along base X and Y and watch
    how a STILL object shifts in the image. Replaces CAM_YAW_DEG / CAM_FLIP / CAM_HFOV_DEG."""
    if isinstance(cam, SimCamera):
        cam.paused, was_manual = True, cam.manual
        cam.manual = cam.object_xy()

    def measure(n=8):
        pts, last = [], None
        t_end = time.time() + 3.0
        while len(pts) < n and time.time() < t_end:
            frame, seq = cam.read(rtde_r.getActualTCPPose(), time.time())
            if seq != last:
                last, d = seq, detect(frame, color)
                if d is not None:
                    pts.append(d[:2])
            time.sleep(0.01)
        return np.mean(pts, axis=0) if len(pts) >= n // 2 else None

    def go(pose):
        rtde_c.moveL(list(pose), 0.03, 0.2)
        time.sleep(0.4)          # let the camera's frames catch up with the robot

    print(f'Calibrating: the object must stay still, the TCP moves +-{step * 100:.0f} cm in X and Y...')
    J = np.zeros((2, 2))
    try:
        if measure() is None:
            raise RuntimeError('object not visible at the start pose')
        for axis in (0, 1):
            seen = []
            for sign in (+1, -1):
                p = list(pose0)
                p[axis] += sign * step
                go(p)
                m = measure()
                if m is None:
                    raise RuntimeError(f'object lost after moving {"XY"[axis]} {sign * step:+.2f} m')
                seen.append(m)
            go(pose0)
            J[:, axis] = (seen[0] - seen[1]) / (2 * step)
    finally:
        go(pose0)
        if isinstance(cam, SimCamera):
            cam.paused, cam.manual = False, was_manual
    cols = np.linalg.norm(J, axis=0)
    if cols.min() < 200 or abs(np.linalg.det(J)) < 0.3 * cols.prod():
        raise RuntimeError(f'Calibration unreliable (J = {J.round(0).tolist()} px/m): '
                           'object too small / too far, or it moved')
    # describe J as scale + rotation + mirror, to compare with the mounting you expect
    mirrored = np.linalg.det(J) > 0      # a down-looking camera without mirror gives det < 0 here
    print(f'Image Jacobian (px per m): {J.round(0).tolist()}')
    print(f'  ~{1000 / cols.mean():.2f} mm per pixel at this height, '
          f'image x axis = base ({J[0, 0] / cols[0]:+.2f}, {J[0, 1] / cols[1]:+.2f}) direction, '
          f'{"mirrored" if mirrored else "not mirrored"}')
    return J


# ------------------------------------------------------------------ main loop
def draw_overlay(img, det, K, follower, v, status):
    h, w = img.shape[:2]
    cx, cy = int(K[0, 2]), int(K[1, 2])
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    if det is not None:
        p = (int(det[0]), int(det[1]))
        cv2.circle(img, p, 8, (255, 255, 255), 2)
        cv2.line(img, (cx, cy), p, (255, 255, 255), 1)
    cv2.putText(img, f'{status}  v=({v[0]*100:+.1f}, {v[1]*100:+.1f}) cm/s', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    global GAIN, MAX_SPEED, FOLLOW_Z, FEEDFORWARD, WORKSPACE
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sim', action='store_true', help='synthetic tool camera (closed loop with URSim)')
    ap.add_argument('--webcam-table', action='store_true',
                    help='PC webcam image laid on the virtual table, seen by the tool camera (closed loop with URSim)')
    ap.add_argument('--no-robot', action='store_true', help='do not connect to the robot')
    ap.add_argument('--camera', default='0',
                    help="index, device path, or 'realsense' (finds the RealSense colour stream)")
    ap.add_argument('--color', default='red', choices=list(COLOR_RANGES))
    ap.add_argument('--ip', default=ROBOT_IP)
    ap.add_argument('--gain', type=float, default=GAIN)
    ap.add_argument('--feedforward', type=float, default=FEEDFORWARD)
    ap.add_argument('--max-speed', type=float, default=MAX_SPEED)
    ap.add_argument('--height', type=float, default=FOLLOW_Z)
    ap.add_argument('--start-here', action='store_true',
                    help='real robot: follow from the CURRENT pose (height + orientation), no move to a preset pose')
    ap.add_argument('--radius', type=float, default=0.15,
                    help='with --start-here: half-size (m) of the XY box around the start pose')
    ap.add_argument('--calibrate', action='store_true',
                    help='measure the camera->robot mapping first (object must be visible and still)')
    ap.add_argument('--sim-mount', type=float, nargs=2, metavar=('YAW', 'FLIP'),
                    help='--sim only: true camera mounting (deg, 0/1), to test --calibrate')
    ap.add_argument('--headless', action='store_true', help='no window (for tests)')
    ap.add_argument('--duration', type=float, default=0, help='stop after N seconds (0 = run until q)')
    args = ap.parse_args()
    GAIN, MAX_SPEED, FOLLOW_Z, FEEDFORWARD = args.gain, min(args.max_speed, 0.25), args.height, args.feedforward
    if (args.sim or args.webcam_table) and args.no_robot:
        sys.exit('--sim / --webcam-table need the robot pose; drop --no-robot')

    local = args.ip in ('127.0.0.1', 'localhost')
    if not local and not args.no_robot and not args.start_here:
        sys.exit('Real robot: use --start-here (jog the camera above the table first). '
                 'The preset start pose and workspace are for URSim only.')

    rtde_c = rtde_r = None
    start = [*START_XY, FOLLOW_Z, *TOOL_DOWN]
    if not args.no_robot:
        rtde_c, rtde_r = connect(args.ip)
        if args.start_here:
            start = list(rtde_r.getActualTCPPose())
            x, y, r = start[0], start[1], args.radius
            WORKSPACE = {'x': (x - r, x + r), 'y': (y - r, y + r)}
            print(f'Following from the current pose: z={start[2]:.3f} m, '
                  f'X in [{x - r:+.3f}, {x + r:+.3f}], Y in [{y - r:+.3f}, {y + r:+.3f}]')
        else:
            seed = [math.radians(a) for a in HOME_SEED_DEG]
            q = rtde_c.getInverseKinematics(start, seed)
            if not q or max(abs(a - b) for a, b in zip(q, seed)) > math.radians(60):
                raise RuntimeError(f'No elbow-up IK solution near HOME_SEED_DEG for {start}')
            print(f'Moving (moveJ) to start pose {START_XY} at z={FOLLOW_Z} m...')
            rtde_c.moveJ(q, 0.8, 1.0)
    hold = start[2:]          # z, rx, ry, rz kept constant while following

    if args.webcam_table:
        cam = WebcamTable(args.camera, args.color)
    else:
        cam = SimCamera(args.color, mount=args.sim_mount) if args.sim else Webcam(args.camera)
    window = 'Follow object (q to quit)'
    state = {'pose': start}
    if not args.headless:
        cv2.namedWindow(window)
        if args.sim:
            def on_mouse(event, x, y, flags, _):
                if event == cv2.EVENT_LBUTTONDOWN:
                    cam.manual = cam.pixel_to_table(x, y, state['pose'])
            cv2.setMouseCallback(window, on_mouse)

    follower, K = Follower(), None
    if args.calibrate:
        if not rtde_c:
            sys.exit('--calibrate needs the robot')
        try:
            follower.J = calibrate(rtde_c, rtde_r, cam, args.color, start)
        except Exception:
            rtde_c.stopScript(); cam.close()
            raise
    target = np.array(start[:2], dtype=float)
    last_seq, det, frame = 0, None, None
    errors, t0, last_print, tick = [], time.time(), 0.0, 0
    try:
        while True:
            t_start = rtde_c.initPeriod() if rtde_c else None
            now = time.time() - t0
            pose = rtde_r.getActualTCPPose() if rtde_r else state['pose']
            state['pose'] = pose
            if rtde_r and robot_fault(rtde_r):
                print('Robot fault (protective / emergency stop): stopping'); break

            frame, seq = cam.read(pose, now)
            if K is None:
                K = intrinsics(frame.shape[1], frame.shape[0])
            new = seq != last_seq
            if new:
                det, last_seq = detect(frame, args.color), seq
            v = follower.update(det if new else None, K, pose, now)

            target = np.array(clamp_xy(*(target + v * DT)))
            if rtde_c:
                rtde_c.servoL([*target, *hold], 0.0, 0.0, DT, 0.1, 300)

            if args.sim:  # ground-truth error: object vs. point under the camera
                below = cam.pixel_to_table(K[0, 2], K[1, 2], pose)
                errors.append((now, np.linalg.norm(below - cam.object_xy())))

            status = 'TRACKING' if follower.tracking else 'LOST'
            if now - last_print > 1.0:
                last_print = now
                extra = f'  err {errors[-1][1]*1000:5.1f} mm' if errors else ''
                print(f't={now:5.1f}s {status:8s} tcp=({pose[0]:+.3f}, {pose[1]:+.3f}) '
                      f'v=({v[0]:+.3f}, {v[1]:+.3f}) m/s{extra}')

            tick += 1
            if not args.headless and tick % 2 == 0:   # display at 25 Hz
                shown = draw_overlay(frame.copy(), det, K, follower, v, status)
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
            if args.duration and now > args.duration:
                break
            if rtde_c:
                rtde_c.waitPeriod(t_start)
            else:
                time.sleep(max(0.0, DT - (time.time() - t0 - now)))
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
        cv2.destroyAllWindows()

    if errors:
        settled = [e for t, e in errors if t > 5.0] or [e for _, e in errors]
        print(f'Tracking error after 5 s: mean {np.mean(settled)*1000:.1f} mm, '
              f'max {np.max(settled)*1000:.1f} mm ({len(settled)} samples)')


if __name__ == '__main__':
    main()
