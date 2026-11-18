"""OpenCV webcam and Intel RealSense (colour + aligned depth) for follow_object."""
import sys
import threading
import time

import cv2
import numpy as np

DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 2.0


def resolve_camera(name):
    """'0' -> 0, 'realsense' -> the RealSense colour stream (OpenCV), anything else -> device path."""
    name = str(name)
    if name.isdigit():
        return int(name)
    if name.lower() == 'realsense':
        if sys.platform == 'darwin':
            return realsense_index_macos()
        # by-id names include the serial number only when the camera reports it, so match loosely;
        # video-index0 is the colour stream (index2 is infrared)
        import glob
        paths = sorted(glob.glob('/dev/v4l/by-id/*RealSense*-video-index0'))
        if not paths:
            raise RuntimeError('No RealSense found under /dev/v4l/by-id (check the USB cable)')
        return paths[0]
    return name


def realsense_index_macos():
    """OpenCV camera index of the RealSense on macOS (AVFoundation list, prefer RGB)."""
    try:
        import AVFoundation
    except ImportError:
        raise RuntimeError('--camera realsense on macOS needs pyobjc-framework-AVFoundation '
                           '(installed by `uv sync`), or pass the camera number, e.g. --camera 1') from None
    devices = (list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
               + list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeMuxed)))
    devices.sort(key=lambda d: str(d.uniqueID()))
    names = [str(d.localizedName()) for d in devices]
    found = [i for i, n in enumerate(names) if 'realsense' in n.lower()]
    if not found:
        raise RuntimeError(f'No RealSense among the cameras: {names} (check the USB cable)')
    rgb = [i for i in found if 'rgb' in names[i].lower() or 'color' in names[i].lower()]
    index = (rgb or found)[0]
    print(f'RealSense (OpenCV): camera {index} ({names[index]})')
    return index


def open_camera(name):
    """'realsense' -> RealSenseCamera (librealsense); otherwise OpenCV Webcam."""
    if str(name).lower() == 'realsense':
        return RealSenseCamera()
    return Webcam(name)


class Webcam:
    """Grabs frames in a thread so the 50 Hz control loop never blocks on the camera."""

    def __init__(self, index):
        index = resolve_camera(index)
        # V4L2 is Linux-only; on macOS use AVFoundation (OpenCV default backend).
        if sys.platform.startswith('linux'):
            self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f'Cannot open camera {index}. '
                'On macOS: allow Camera for Terminal/Cursor, or try --camera 0 / --sim'
            )
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


class RealSenseCamera:
    """D435i colour + depth via pyrealsense2, depth aligned to colour.

    Same public shape as Webcam: threaded grab, read(tcp_pose, now) -> (bgr, seq), close().
    depth_at(u, v) returns metres (median in a small window), or None if no valid samples.
    """

    def __init__(self, width=640, height=480, fps=30):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                'RealSense needs pyrealsense2 (Linux/Windows) or pyrealsense2-macosx (macOS); '
                'run `uv sync`'
            ) from e
        self._rs = rs
        self.pipeline = None
        self.running = False
        print('Opening RealSense (librealsense colour+depth)...', flush=True)
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if devices.size() == 0:
                raise RuntimeError('No RealSense device found (check the USB cable, prefer USB 3)')
            dev = devices[0]
            print(f'  found: {dev.get_info(rs.camera_info.name)} '
                  f'sn={dev.get_info(rs.camera_info.serial_number)}', flush=True)
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            self._safe_stop()
            hint = ''
            if 'power state' in str(e).lower() or sys.platform == 'darwin':
                hint = (' On macOS, librealsense often fails with "failed to set power state": '
                        'try another USB-C/USB 3 port, unplug/replug, or run on the lab Linux PC. '
                        'sudo does not always fix it.')
            raise RuntimeError(f'Cannot start RealSense: {e}.{hint}') from e
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.frame = None
        self.depth = None          # float32 metres, same size as colour, 0 = invalid
        self.seq, self.running = 0, True
        self.lock = threading.Lock()
        # block until the first aligned pair is ready
        self._grab_once()
        if self.frame is None:
            self._safe_stop()
            raise RuntimeError('RealSense produced no frames')
        print(f'RealSense: colour+depth {width}x{height} @ {fps} Hz '
              f'(depth scale {self.depth_scale * 1000:.2f} mm/unit)', flush=True)
        threading.Thread(target=self._loop, daemon=True).start()

    def _safe_stop(self):
        pipe, self.pipeline = self.pipeline, None
        self.running = False
        if pipe is not None:
            try:
                pipe.stop()
            except Exception:
                pass

    def _grab_once(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return
        bgr = np.asanyarray(color.get_data())
        z = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        with self.lock:
            self.frame, self.depth, self.seq = bgr, z, self.seq + 1

    def _loop(self):
        while self.running:
            self._grab_once()

    def read(self, tcp_pose, now):
        with self.lock:
            return self.frame, self.seq

    def depth_at(self, u, v, window=5):
        """Median valid depth (m) around pixel (u, v), or None."""
        with self.lock:
            depth = None if self.depth is None else self.depth.copy()
        if depth is None:
            return None
        h, w = depth.shape
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < w and 0 <= y < h):
            return None
        r = max(window // 2, 0)
        patch = depth[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        valid = patch[(patch >= DEPTH_MIN_M) & (patch <= DEPTH_MAX_M)]
        return float(np.median(valid)) if valid.size else None

    def close(self):
        self.running = False
        time.sleep(0.05)
        self._safe_stop()
