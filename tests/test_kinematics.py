import math

import numpy as np
import pytest

from cobot.kinematics import forward_kinematics, matrix_to_pose, pose_to_matrix


def test_zero_configuration():
    # All joints at 0: arm stretched along -X, flange offset by d4 + d6 along -Y.
    x, y, z = forward_kinematics([0] * 6)[:3]
    assert x == pytest.approx(-0.425 - 0.3922)
    assert y == pytest.approx(-(0.1333 + 0.0996))
    assert z == pytest.approx(0.1625 - 0.0997)


def test_base_rotation_rotates_position():
    p0 = np.array(forward_kinematics([0, -1.2, 1.0, -1.4, -1.57, 0.3])[:3])
    p1 = np.array(forward_kinematics([math.pi / 2, -1.2, 1.0, -1.4, -1.57, 0.3])[:3])
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(p1, rz @ p0, atol=1e-12)


def test_tcp_offset_moves_along_tool_z():
    q = [0.3, -1.2, 1.0, -1.4, -1.57, 0.3]
    flange = pose_to_matrix(forward_kinematics(q))
    tcp = forward_kinematics(q, tcp_offset=(0, 0, 0.1, 0, 0, 0))
    np.testing.assert_allclose(tcp[:3], flange[:3, 3] + 0.1 * flange[:3, 2], atol=1e-12)


def test_pose_matrix_round_trip():
    pose = [0.1, -0.2, 0.3, 0.4, -1.2, 2.0]
    np.testing.assert_allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-12)


def test_wrong_length():
    with pytest.raises(ValueError):
        forward_kinematics([0] * 5)


@pytest.mark.ursim
def test_matches_ursim():
    """FK of the current joints matches the controller's TCP pose (needs URSim)."""
    rtde_receive = pytest.importorskip("rtde_receive")
    try:
        r = rtde_receive.RTDEReceiveInterface("127.0.0.1")
    except RuntimeError:
        pytest.skip("URSim not reachable")
    try:
        expected = pose_to_matrix(r.getActualTCPPose())
        actual = pose_to_matrix(forward_kinematics(r.getActualQ()))
    finally:
        r.disconnect()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
