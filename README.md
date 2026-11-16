# Cobotique: UR5e project

Code for controlling a Universal Robots UR5e from Python, in the URSim simulator: forward kinematics, safe
joint and linear moves, camera to robot calibration and colour-based picking. It builds on the
"IA et Cobotique" workshop at Telecom Nancy. The workshop's Day 1 and Day 2 notebooks are kept at the repository root.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Docker, git-lfs (`sudo dnf install git-lfs`) and a webcam for the vision code.

```bash
uv sync                      # .venv with Python 3.12 (ur_rtde has no wheels for newer versions)
scripts/setup_git.sh         # once per clone: notebook output stripping + Git LFS
scripts/download_models.sh   # MobileNet-SSD weights (checksummed, not in git)
ursim/bringup.sh             # start URSim, power on, release brakes
uv run pytest                # tests; the `ursim` one needs URSim running (skip with -m "not ursim")
```

Polyscope GUI: http://localhost:6080/vnc.html. Remote Control is already enabled through
`ursim/polyscope/remotecontrol.properties`. Stop URSim with `docker compose down`.

## Layout

| Path | What |
|---|---|
| `src/cobot/` | Shared package, installed editable by `uv sync` (`import cobot`) |
| `src/cobot/robot.py` | `Robot`: an RTDE connection with checked `move_j`, `move_l` and `gripper` |
| `src/cobot/kinematics.py` | UR5e forward kinematics (DH), pose ↔ matrix |
| `src/cobot/vision.py` | Camera helpers, HSV colour detection, MobileNet-SSD |
| `src/cobot/calibration.py` | Checkerboard corner ordering, camera → robot homography, `pixel_to_robot` |
| `config/calibration.json` | Measured board corners A/B/C, square size and camera index for **your** table |
| `data/` | Calibration images and `homography.npy` (git-ignored; specific to each setup) |
| `tests/` | pytest suite |
| `compose.yaml`, `ursim/` | URSim 5.26 container (image pinned by digest), programs and URCaps volumes |
| `Day*_*.ipynb`, `*.pptx` | Workshop notebooks and lecture decks (decks stored in Git LFS) |

## Usage

```python
from cobot import Robot, TOOL_DOWN
from cobot.calibration import load_config, load_homography, pixel_to_robot
from cobot.vision import grab_frame, detect_colored_objects

cfg = load_config()
H = load_homography(cfg)
objects = detect_colored_objects(grab_frame(cfg.camera_index))

with Robot() as robot:                       # 127.0.0.1 by default
    for obj in objects:
        x, y = pixel_to_robot(*obj["center"], H)
        robot.move_l([x, y, 0.15] + TOOL_DOWN)
```

Every move is checked before it runs (robot state, speed and acceleration, joint limits, the `Limits.workspace` box, and the controller
safety limits) and verified afterwards. A move returns `False` and prints the reason if a check fails.

## Gotchas

- **Only one `RTDEControlInterface` at a time.** `One of the RTDE input registers are already in use!` means
  another kernel or script still holds it. Use `Robot` as a context manager, or call `robot.close()`, or restart the
  kernel. Any number of `RTDEReceiveInterface` readers is fine.
- `Failed to start RTDE data synchronization` means Remote Control is off in Polyscope.
- Don't use `rtde_c.getForwardKinematics(q)` with an explicit `q`: on ur_rtde 1.6.5 with Polyscope 5.26 it
  returns wrong poses. Use `cobot.forward_kinematics` instead (tested against the live controller).
- Calibration: keep the A/B/C three-corner scheme. A wrong board orientation can still fit with
  near-zero reprojection error and send the robot 40 cm off (see `cobot/calibration.py`).
- OpenCV is pinned `<5` because OpenCV 5 removed `cv2.dnn.readNetFromCaffe`.
