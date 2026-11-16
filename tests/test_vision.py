import cv2
import numpy as np

from cobot.vision import detect_colored_objects


def test_detects_red_and_blue_blobs():
    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 30), (100, 110), (0, 0, 255), -1)     # red, 81x81
    cv2.rectangle(img, (200, 150), (240, 190), (255, 0, 0), -1)   # blue, 41x41
    objects = detect_colored_objects(img)
    assert [o["color"] for o in objects] == ["red", "blue"]        # largest first
    np.testing.assert_allclose(objects[0]["center"], (60, 70), atol=1.5)
    np.testing.assert_allclose(objects[1]["center"], (220, 170), atol=1.5)


def test_ignores_small_blobs():
    img = np.full((100, 100, 3), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (20, 20), (0, 0, 255), -1)
    assert detect_colored_objects(img) == []
