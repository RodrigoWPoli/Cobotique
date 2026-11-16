"""Camera -> robot table-plane calibration with a checkerboard (from Day 2).

The robot (X, Y) of three board corners A, B, C are measured and stored in
config/calibration.json. ``find_board`` re-orders detected corners so that A is always the image's
top-left corner. Do not simplify this to origin + angle: the detector can start from either end of
the board, and a downward camera mirrors the table, so a wrong orientation still fits with
near-zero reprojection error but sends the robot 40 cm off.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cobot import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "calibration.json"


@dataclass
class CalibrationConfig:
    camera_index: int
    board_inner: tuple        # inner corners (per row, per column)
    square_size: float        # m
    corner_a: tuple           # robot (X, Y) m of corner A (image top-left)
    corner_b: tuple           # A + (cols-1) squares along the row
    corner_c: tuple           # A + (rows-1) squares along the column
    homography_file: str

    @property
    def homography_path(self):
        p = Path(self.homography_file)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path=DEFAULT_CONFIG):
    data = json.loads(Path(path).read_text())
    for key in ("board_inner", "corner_a", "corner_b", "corner_c"):
        data[key] = tuple(data[key])
    return CalibrationConfig(**data)


def find_board(img, cfg):
    """(N, 2) inner-corner pixels ordered from corner A (image top-left), or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found, corners = cv2.findChessboardCornersSB(gray, cfg.board_inner)
    if not found:
        return None
    cols, rows = cfg.board_inner
    grid = corners.reshape(rows, cols, 2)
    options = [grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1]]
    grid = min(options, key=lambda g: g[0, 0].sum())
    return grid.reshape(-1, 2)


def board_points_robot(cfg):
    """Robot (X, Y) of every inner corner, in the same order as find_board()."""
    cols, rows = cfg.board_inner
    a, b, c = (np.array(p, dtype=float) for p in (cfg.corner_a, cfg.corner_b, cfg.corner_c))
    step_row, step_col = (b - a) / (cols - 1), (c - a) / (rows - 1)
    i, j = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).T
    return a + np.outer(i, step_row) + np.outer(j, step_col)


def check_measurements(cfg, tolerance=0.005):
    """True if |AB|, |AC| match the square size and angle A is ~90°."""
    cols, rows = cfg.board_inner
    a, b, c = (np.array(p, dtype=float) for p in (cfg.corner_a, cfg.corner_b, cfg.corner_c))
    ab, ac = np.linalg.norm(b - a), np.linalg.norm(c - a)
    angle = math.degrees(math.acos(np.dot(b - a, c - a) / (ab * ac)))
    return bool(abs(ab - (cols - 1) * cfg.square_size) < tolerance
                and abs(ac - (rows - 1) * cfg.square_size) < tolerance
                and abs(angle - 90) < 3)


def compute_homography(images, cfg, ransac_threshold=0.005):
    """Fit the pixel -> robot (X, Y) homography from calibration images.

    Returns (H, errors_mm), errors_mm being the per-corner reprojection error.
    """
    pixels, world = [], []
    for img in images:
        corners = find_board(img, cfg)
        if corners is not None:
            pixels.append(corners)
            world.append(board_points_robot(cfg))
    if not pixels:
        raise RuntimeError("board not found in any calibration image")
    pixels = np.vstack(pixels).astype(np.float64)
    world = np.vstack(world).astype(np.float64)
    H, _ = cv2.findHomography(pixels, world, cv2.RANSAC, ransac_threshold)
    mapped = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, np.linalg.norm(mapped - world, axis=1) * 1000


def save_homography(H, cfg):
    cfg.homography_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.homography_path, H)


def load_homography(cfg):
    return np.load(cfg.homography_path)


def pixel_to_robot(px, py, H):
    """Pixel (px, py) -> robot (X, Y) in metres, on the table plane."""
    X, Y = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float64), H)[0, 0]
    return float(X), float(Y)
