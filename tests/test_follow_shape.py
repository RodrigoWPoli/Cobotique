"""Shape matching on top of the colour blob detector in project/detection.py."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'project'))
import detection as det  # noqa: E402

BLUE = (255, 0, 0)  # BGR


def _learn_rectangle(tmp_path):
    ref = np.full((220, 240, 3), 255, np.uint8)
    cv2.rectangle(ref, (50, 40), (170, 170), BLUE, -1)   # ~1.08 aspect, filled
    path = tmp_path / 'ref.png'
    cv2.imwrite(str(path), ref)
    return det.learn_shape(str(path), 'blue')


def test_shape_prefers_the_rectangle_over_a_larger_circle(tmp_path):
    model = _learn_rectangle(tmp_path)
    scene = np.full((400, 640, 3), 255, np.uint8)
    cv2.circle(scene, (120, 200), 110, BLUE, -1)                 # larger area
    cv2.rectangle(scene, (400, 160), (530, 300), BLUE, -1)       # same shape, smaller
    cv2.rectangle(scene, (360, 20), (620, 80), BLUE, -1)         # long bar

    colour_only = det.detect(scene, 'blue')
    np.testing.assert_allclose(colour_only[:2], (120, 200), atol=4)

    found = det.detect(scene, 'blue', model)
    assert found is not None and found[4] <= det.SHAPE_MAX
    np.testing.assert_allclose(found[:2], (465, 230), atol=4)


def test_shape_rejects_circle_and_keeps_a_rotated_copy(tmp_path):
    model = _learn_rectangle(tmp_path)
    only_circle = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(only_circle, (150, 150), 80, BLUE, -1)
    assert det.detect(only_circle, 'blue', model) is None

    turned = np.full((360, 360, 3), 255, np.uint8)
    box = cv2.boxPoints(((180, 180), (140, 120), 35)).astype(np.int32)
    cv2.fillPoly(turned, [box], BLUE)
    found = det.detect(turned, 'blue', model)
    assert found is not None
    np.testing.assert_allclose(found[:2], (180, 180), atol=6)


def test_capture_shape_learns_the_frame_and_saves_it(tmp_path):
    frame = np.full((240, 320, 3), 255, np.uint8)
    cv2.rectangle(frame, (60, 50), (180, 170), BLUE, -1)
    model, path = det.capture_shape(frame, 'blue', folder=str(tmp_path))
    assert os.path.dirname(path) == str(tmp_path) and os.path.basename(path).startswith('shape_')
    assert det.resolve_shape_photo(path) == path
    reloaded = det.learn_shape(path, 'blue')          # the saved photo gives the same silhouette
    assert abs(reloaded.aspect - model.aspect) < 1e-6

    empty = np.full((240, 320, 3), 255, np.uint8)
    with pytest.raises(RuntimeError):
        det.capture_shape(empty, 'blue', folder=str(tmp_path / 'none'))
    assert not (tmp_path / 'none').exists()           # nothing saved when nothing was learned


def test_reference_photo_is_the_square_bin_not_a_circle():
    # The lab's bin photo. Not "the newest photo in data/": the s key saves new captures there.
    path = 'data/shape_ref.jpg'
    if not os.path.exists(path):
        pytest.skip('no reference photo data/shape_ref.jpg')
    model = det.learn_shape(path, 'blue')
    assert 1.0 <= model.aspect <= 1.3
    assert model.solidity > 0.95

    photo = cv2.imread(path)
    on_photo = det.detect(photo, 'blue', model)
    assert on_photo is not None
    # the bin sits in the upper middle of the picture, not on the sweater
    h, w = photo.shape[:2]
    assert w * 0.4 < on_photo[0] < w * 0.7
    assert on_photo[1] < h * 0.55

    scene = np.full_like(photo, 255)
    cv2.drawContours(scene, [model.contour], -1, BLUE, -1)
    cv2.circle(scene, (260, 700), 220, BLUE, -1)   # bigger than the bin, and clear of it
    found = det.detect(scene, 'blue', model)
    colour_only = det.detect(scene, 'blue')
    assert found is not None
    np.testing.assert_allclose(found[:2], on_photo[:2], atol=8)
    assert abs(colour_only[0] - 260) < 8          # colour alone follows the circle
