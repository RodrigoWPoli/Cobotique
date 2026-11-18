"""Colour and optional silhouette detection for follow_object.

HSV blob detection (same ranges as Day 2), plus optional matching against a reference
silhouette learned from a photo (--shape).
"""
import glob
import math
import os

import cv2
import numpy as np

# HSV ranges (OpenCV: H 0..179); red wraps around so it needs two ranges
COLOR_RANGES = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
    'green':  [((40, 80, 50), (85, 255, 255))],
    'blue':   [((100, 150, 50), (130, 255, 255))],
}
DRAW_BGR = {'red': (0, 0, 255), 'yellow': (0, 220, 220), 'green': (0, 200, 0), 'blue': (255, 0, 0)}
MIN_AREA = 300                          # px²
# Worst silhouette feature (aspect, fill, circularity, Hu moments), in units of "still this
# object". 1 accepts a turned rectangle and rejects a circle, a long bar, or an irregular blob.
SHAPE_MAX = 1.0


def color_mask(img, color):
    """Binary mask of `color` after a small open/close, same ranges as Day 2."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _silhouette(contour):
    """Scale- and rotation-invariant outline: aspect, circularity, solidity, fill of the
    minimum-area rectangle (an axis-aligned box would change when the object rolls)."""
    area = float(cv2.contourArea(contour))
    peri = max(float(cv2.arcLength(contour, True)), 1e-6)
    rw, rh = cv2.minAreaRect(contour)[1]
    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
    solidity = area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1e-6)
    rect_fill = area / max(rw * rh, 1e-6)
    circularity = 4 * math.pi * area / (peri * peri)
    return aspect, circularity, solidity, rect_fill


class ShapeModel:
    """Silhouette learned from a reference photo. Compared live with matchShapes (Hu moments)
    plus how rectangular and how filled the blob is. Independent of size and in-plane rotation."""

    def __init__(self, contour):
        self.contour = contour
        self.aspect, self.circularity, self.solidity, self.rect_fill = _silhouette(contour)

    def distance(self, contour):
        """0 = same outline. Each term reaches 1 at the edge of 'still this object'."""
        aspect, circularity, solidity, rect_fill = _silhouette(contour)
        d_aspect = abs(math.log(aspect / self.aspect)) / math.log(1.45)
        d_circ = abs(circularity - self.circularity) / 0.12
        d_sol = abs(solidity - self.solidity) / 0.12
        d_fill = abs(rect_fill - self.rect_fill) / 0.15
        d_hu = cv2.matchShapes(self.contour, contour, cv2.CONTOURS_MATCH_I1, 0.0) / 0.15
        return max(d_aspect, d_circ, d_sol, d_fill, d_hu)


def resolve_shape_photo(spec):
    """`--shape` with no path: the newest image in data/."""
    if spec not in (None, 'auto'):
        return spec
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        files.extend(glob.glob(os.path.join('data', ext)))
    if not files:
        raise RuntimeError('no photo in data/: pass --shape path/to/photo.jpg')
    return max(files, key=os.path.getmtime)


def learn_shape(path, color):
    """Largest blob of `color` in the photo. That blob should be the object, and the biggest
    region of that colour in the picture."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    contours, _ = cv2.findContours(color_mask(img, color), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea, default=None)
    if best is None or cv2.contourArea(best) < MIN_AREA:
        raise RuntimeError(f'no {color} object large enough in {path}')
    model = ShapeModel(best)
    m = cv2.moments(best)
    print(f'Shape learned from {path}: {color}, aspect {model.aspect:.2f}, '
          f'filled {model.solidity:.2f}, area {cv2.contourArea(best):.0f} px, '
          f'centre ({m["m10"] / m["m00"]:.0f}, {m["m01"] / m["m00"]:.0f}). '
          f'A live blob is kept only while its silhouette stays close to this one '
          f'(distance <= {SHAPE_MAX:.1f}).')
    return model


def detect(img, color, shape=None):
    """Blob of `color` to follow: (cx, cy, area, contour, distance) in pixels, or None.

    Without `shape`, the largest blob. With a ShapeModel, the blob whose silhouette is closest
    to the reference, and only if that distance is <= SHAPE_MAX. `distance` is None without a model.
    """
    contours, _ = cv2.findContours(color_mask(img, color), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    distance = None
    if shape is not None:
        scored = []
        for c in candidates:
            d = shape.distance(c)
            if d <= SHAPE_MAX:
                scored.append((d, c))
        if not scored:
            return None
        distance, best = min(scored, key=lambda t: (t[0], -cv2.contourArea(t[1])))
    else:
        best = max(candidates, key=cv2.contourArea, default=None)
        if best is None:
            return None
    m = cv2.moments(best)
    if m['m00'] == 0:
        return None
    return m['m10'] / m['m00'], m['m01'] / m['m00'], m['m00'], best, distance
