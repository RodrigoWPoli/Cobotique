"""Shared code for the cobot project: UR5e kinematics, safe robot moves, vision and calibration.

    from cobot import Robot, TOOL_DOWN, forward_kinematics
    from cobot.calibration import load_config, load_homography, pixel_to_robot
    from cobot.vision import detect_colored_objects
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from cobot.kinematics import forward_kinematics, matrix_to_pose, pose_to_matrix  # noqa: E402
from cobot.robot import TOOL_DOWN, Limits, Robot  # noqa: E402

__all__ = ["PROJECT_ROOT", "TOOL_DOWN", "Limits", "Robot",
           "forward_kinematics", "matrix_to_pose", "pose_to_matrix"]
