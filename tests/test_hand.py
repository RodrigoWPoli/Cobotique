"""Hand open/closed logic in project/hand.py and the --hand return-home velocity."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'project'))
import follow_object as fo  # noqa: E402
import hand as hd  # noqa: E402


def _hand(curl, angle_deg=0.0):
    """Synthetic 21 landmarks (pixels). Fingers point up the image; curl=True folds them onto the palm.
    The whole hand is rotated by angle_deg about the wrist."""
    p = np.zeros((21, 2))
    p[0] = (0, 0)
    p[1:5] = [(-30, -20), (-50, -40), (-65, -55), (-75, -70)]       # thumb
    for k, x in enumerate([-30, -10, 10, 30]):                      # index .. little
        mcp, pip, dip, tip = 5 + 4 * k, 6 + 4 * k, 7 + 4 * k, 8 + 4 * k
        p[mcp] = (x, -80)
        if curl:   # PIP sticks out, DIP and tip come back toward the palm
            p[pip], p[dip], p[tip] = (x, -105), (x, -85), (x, -65)
        else:
            p[pip], p[dip], p[tip] = (x, -115), (x, -140), (x, -160)
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return p @ R.T + (320, 400)


@pytest.mark.parametrize('angle', [0, 70, 180, -120])
def test_finger_count_open_and_fist_any_rotation(angle):
    assert hd.fingers_extended(_hand(curl=False, angle_deg=angle)) == 4
    assert hd.fingers_extended(_hand(curl=True, angle_deg=angle)) == 0


def test_classify_combines_label_and_fingers():
    assert hd.classify('Open_Palm', 2) == 'open'
    assert hd.classify(None, 4) == 'open'
    assert hd.classify('Closed_Fist', 3) == 'closed'
    assert hd.classify('None', 0) == 'closed'
    assert hd.classify('Pointing_Up', 1) == 'uncertain'


def test_hand_detection_tuple_is_the_palm_centre():
    h = hd.Hand(_hand(curl=False), 'Open_Palm', 0.9)
    cx, cy, area, contour, dist = h.as_detection()
    np.testing.assert_allclose((cx, cy), h.landmarks[hd.PALM].mean(axis=0))
    assert area > 0 and contour is None and dist is None


def test_gate_follows_any_visible_hand_and_goes_home_only_without_one():
    g = hd.HandGate(present_confirm=0.3, home_delay=0.5)
    assert g.update('closed', 0.0) == 'home'
    assert g.update(None, 0.1) == 'home'          # a flash of hand is not enough
    for t in np.arange(0.2, 0.5, 0.05):
        g.update('closed', t)                     # a fist starts the following too
    assert g.update('closed', 0.55) == 'follow'
    for t, state in [(0.7, 'open'), (0.9, 'closed'), (1.1, 'uncertain'), (1.3, 'closed')]:
        assert g.update(state, t) == 'follow'     # open, fist or unclear: still following
    assert g.update(None, 1.6) == 'follow'        # brief loss: keep following
    assert g.update(None, 1.85) == 'home'         # 0.55 s without a hand


def test_uncertain_frames_do_not_break_a_fist_being_confirmed():
    g = hd.HandGate(present_confirm=0.0, home_delay=0.5, grip_confirm=0.3)
    g.update('closed', 0.0)
    g.update('uncertain', 0.15)
    g.update('closed', 0.3)
    assert g.grip == 'closed'


def test_home_velocity_returns_to_the_start_within_the_speed_cap():
    target, v = np.array([0.10, -0.05, 0.02]), np.zeros(3)
    for _ in range(int(30 / fo.DT)):
        v = fo.home_velocity(target, v, fo.DT)
        assert np.linalg.norm(v) <= fo.MAX_SPEED + 1e-9
        target = target + v * fo.DT
    assert np.linalg.norm(target) < 2e-3


def test_grip_follows_a_seen_fist_or_open_hand_but_not_a_lost_hand():
    g = hd.HandGate(present_confirm=0.3, home_delay=0.5, grip_confirm=0.3)
    assert g.grip is None
    g.update('closed', 0.0)
    g.update('closed', 0.2)
    assert g.grip is None                  # fist not held long enough yet
    g.update('closed', 0.35)
    assert g.grip == 'closed'
    for t in (0.5, 1.0, 2.0):
        g.update(None, t)                  # hand lost: robot goes home, claw stays closed
    assert g.grip == 'closed' and g.mode == 'home'
    g.update('open', 2.1)
    g.update(None, 2.2)                    # interrupted: the open hand must be seen again for 0.3 s
    g.update('open', 2.3)
    g.update('open', 2.5)
    assert g.grip == 'closed'
    g.update('open', 2.65)
    assert g.grip == 'open'
