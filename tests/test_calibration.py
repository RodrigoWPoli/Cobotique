import cv2
import numpy as np
import pytest

from cobot.calibration import (CalibrationConfig, board_points_robot, check_measurements,
                               compute_homography, find_board, load_config, pixel_to_robot)

SQUARE_PX, MARGIN_PX = 40, 60


@pytest.fixture
def cfg():
    return CalibrationConfig(camera_index=0, board_inner=(9, 6), square_size=0.025,
                             corner_a=(-0.55, -0.35), corner_b=(-0.35, -0.35), corner_c=(-0.55, -0.225),
                             homography_file="data/homography.npy")


def board_image(cols=10, rows=7):
    """Synthetic grey image of a cols x rows squares checkerboard on a white margin."""
    img = np.full((rows * SQUARE_PX + 2 * MARGIN_PX, cols * SQUARE_PX + 2 * MARGIN_PX), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = MARGIN_PX + r * SQUARE_PX, MARGIN_PX + c * SQUARE_PX
                img[y:y + SQUARE_PX, x:x + SQUARE_PX] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_default_config_loads():
    c = load_config()
    assert len(c.board_inner) == 2 and c.square_size > 0


def test_measurements_consistent(cfg):
    assert check_measurements(cfg)
    cfg.corner_b = (-0.30, -0.35)
    assert not check_measurements(cfg)


def test_board_points_hit_a_b_c(cfg):
    pts = board_points_robot(cfg)
    cols, rows = cfg.board_inner
    assert len(pts) == cols * rows
    np.testing.assert_allclose(pts[0], cfg.corner_a)
    np.testing.assert_allclose(pts[cols - 1], cfg.corner_b)
    np.testing.assert_allclose(pts[(rows - 1) * cols], cfg.corner_c)


@pytest.mark.parametrize("transform", [None, cv2.ROTATE_180])
def test_find_board_starts_at_top_left(cfg, transform):
    img = board_image()
    if transform is not None:
        img = cv2.rotate(img, transform)
    corners = find_board(img, cfg)
    assert corners is not None
    first = MARGIN_PX + SQUARE_PX
    np.testing.assert_allclose(corners[0], (first, first), atol=1.0)
    # B is along the image row, C along the image column
    assert corners[cfg.board_inner[0] - 1][0] > corners[0][0] + 100
    assert corners[(cfg.board_inner[1] - 1) * cfg.board_inner[0]][1] > corners[0][1] + 100


def test_homography_maps_corners_to_robot(cfg):
    H, errors = compute_homography([board_image()], cfg)
    assert errors.max() < 1.0   # mm
    first = MARGIN_PX + SQUARE_PX
    np.testing.assert_allclose(pixel_to_robot(first, first, H), cfg.corner_a, atol=1e-3)
    np.testing.assert_allclose(pixel_to_robot(first + 8 * SQUARE_PX, first, H), cfg.corner_b, atol=1e-3)


def test_no_board_raises(cfg):
    with pytest.raises(RuntimeError):
        compute_homography([np.full((200, 200, 3), 255, np.uint8)], cfg)
