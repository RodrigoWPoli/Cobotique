"""UR5e forward kinematics from the standard DH parameters.

ur_rtde 1.6.5 + Polyscope 5.26 returns wrong poses from ``getForwardKinematics(q)`` when ``q``
is given explicitly, so compute FK here instead.
"""

import math

import cv2
import numpy as np

# UR5e standard DH parameters (Universal Robots), metres / radians
DH_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
DH_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)


def _dh(theta, d, a, alpha):
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def pose_to_matrix(pose):
    """UR pose [x, y, z, rx, ry, rz] (rotation vector) -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(pose[3:], dtype=float))[0]
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose(T):
    """4x4 homogeneous matrix -> UR pose [x, y, z, rx, ry, rz]."""
    T = np.asarray(T, dtype=float)
    rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
    return [float(v) for v in T[:3, 3]] + [float(v) for v in rvec]


def forward_kinematics(q, tcp_offset=(0, 0, 0, 0, 0, 0)):
    """Joint angles (6, rad) -> TCP pose [x, y, z, rx, ry, rz] in the base frame.

    ``tcp_offset`` is the TCP set in the Polyscope installation (flange -> TCP), as a UR pose.
    """
    if len(q) != 6:
        raise ValueError(f"expected 6 joint values, got {len(q)}")
    T = np.eye(4)
    for theta, d, a, alpha in zip(q, DH_D, DH_A, DH_ALPHA):
        T = T @ _dh(theta, d, a, alpha)
    return matrix_to_pose(T @ pose_to_matrix(tcp_offset))
