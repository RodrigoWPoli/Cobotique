"""Camera helpers, HSV colour detection and MobileNet-SSD object detection (from Day 2)."""

import os
import time

# OpenCV's bundled Qt only ships the X11 plugin; on Wayland, windows go through XWayland.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cobot import PROJECT_ROOT  # noqa: E402

COLOR_RANGES = {   # name: list of (lower HSV, upper HSV)
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "blue":   [((100, 150, 50), (130, 255, 255))],
}
DRAW_COLORS = {"red": (0, 0, 255), "yellow": (0, 220, 220), "green": (0, 200, 0), "blue": (255, 0, 0)}
MIN_AREA = 500     # px², ignore smaller blobs (noise)


# --- images / camera --------------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"{path} not found")
    return img


def show(img, title="", size=(8, 6)):
    """Display a BGR (colour) or single-channel image inline with matplotlib."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=size)
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def grab_frame(camera_index=0, warmup=5):
    """Return one frame from the camera (skips a few frames so exposure settles)."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    try:
        for _ in range(warmup):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame
    finally:
        cap.release()


def run_live(window, process=None, on_key=None, timeout=120, camera_index=0):
    """Show the live camera feed in an OpenCV window and return the last raw frame.

    process(frame) -> image to display; on_key(key, frame) is called for every key press.
    Stops on q / Esc, when the window is closed, or after ``timeout`` seconds.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    last, t0 = None, time.time()
    try:
        while time.time() - t0 < timeout:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break
            last = frame
            cv2.imshow(window, process(frame.copy()) if process else frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if on_key and key != 255:
                on_key(key, frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return last


# --- colour detection -------------------------------------------------------------------------
def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove small specks
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes


def detect_colored_objects(img, colors=COLOR_RANGES, min_area=MIN_AREA):
    """List of dicts (color, center px, area px², angle deg, bbox), largest first."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    objects = []
    for name, ranges in colors.items():
        contours, _ = cv2.findContours(color_mask(hsv, ranges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            m = cv2.moments(c)
            center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            objects.append(dict(color=name, center=center, area=area,
                                angle=cv2.minAreaRect(c)[2], bbox=cv2.boundingRect(c)))
    return sorted(objects, key=lambda o: -o["area"])


def draw_objects(img, objects):
    for o in objects:
        x, y, w, h = o["bbox"]
        cx, cy = map(int, o["center"])
        col = DRAW_COLORS.get(o["color"], (255, 255, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        cv2.circle(img, (cx, cy), 5, col, -1)
        cv2.putText(img, f"{o['color']} ({cx},{cy})", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


# --- MobileNet-SSD ----------------------------------------------------------------------------
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]


class MobileNetSSD:
    """Caffe MobileNet-SSD (20 VOC classes). Run scripts/download_models.sh first."""

    def __init__(self, prototxt=PROJECT_ROOT / "MobileNetSSD_deploy.prototxt",
                 weights=PROJECT_ROOT / "MobileNetSSD_deploy.caffemodel"):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"{weights} missing: run scripts/download_models.sh")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))

    def detect(self, img, min_confidence=0.4):
        """List of (label, confidence, [x1, y1, x2, y2] px)."""
        h, w = img.shape[:2]
        self.net.setInput(cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5))
        detections = self.net.forward()
        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < min_confidence:
                continue
            box = np.clip(detections[0, 0, i, 3:7], 0, 1) * np.array([w, h, w, h])
            results.append((VOC_CLASSES[int(detections[0, 0, i, 1])], confidence, box.astype(int)))
        return results
