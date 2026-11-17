# Cobotique: UR5e object following

**The UR5e follows a coloured object seen by a camera mounted on its tool** (eye-in-hand, image-based
visual servoing). Each frame, `project/follow_object.py`:

1. detects the object by colour (HSV, the same pipeline as Day 2),
2. measures the object's offset from the image centre,
3. turns that offset into a TCP velocity in the robot base frame: proportional control plus
   feedforward of the object's estimated velocity, with deadband, speed and acceleration limits,
4. integrates the velocity into a target pose and streams it with `servoL` at 50 Hz.

The TCP keeps a fixed height and orientation; only X and Y are servoed. The target is always
clamped to a workspace box. The robot brakes to a stop if the object is lost for 0.5 s and the
script exits on a protective or emergency stop.

The project builds on the "IA et Cobotique" workshop at Telecom Nancy, whose material (decks, notebooks) is
kept locally in the git-ignored `workshop/` folder. Reusable building blocks (kinematics, checked moves,
calibration, vision) are in the [`cobot` package](#the-cobot-package).

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

If URSim refuses connections after a protective stop:

```bash
printf 'unlock protective stop\nquit\n' | nc 127.0.0.1 29999
```

## Commands

All commands run from the repository root.

| Goal | Command |
|---|---|
| Simulated tool camera in URSim (closed loop) | `uv run python project/follow_object.py --sim` |
| Self-test, prints the tracking error | `uv run python project/follow_object.py --sim --headless --duration 25` |
| PC webcam with URSim | `uv run python project/follow_object.py --webcam-table --color red` |
| Check detection only, no robot | `uv run python project/follow_object.py --no-robot --color blue --camera "$RS"` |
| **Real robot** | see [Real robot](#real-robot) |

`$RS` is the RealSense colour stream (see [Cameras](#cameras)):

```bash
RS=/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_143523061441-video-index0
```

**Keys** (in the window): `q` / `Esc` quit · `space` pause the simulated object · click to move the
simulated object (`--sim`).

## Modes

| Mode | What it does | Use with |
|---|---|---|
| *(default)* | Camera from `--camera`. The loop only closes when the camera really moves with the tool. | Real robot |
| `--sim` | Renders what a tool camera would see from URSim's actual TCP pose: a 5 cm table grid and a disc moving on a figure-eight. | URSim |
| `--webcam-table` | Lays the (mirrored) webcam image flat on the virtual table, 1.8× larger than the tool camera's view. The tool camera sees the part under the TCP. Move an object in front of the webcam and the robot goes after it. An inset shows the webcam with the tool camera's footprint in yellow. | URSim + PC webcam |
| `--no-robot` | Detection and computed velocity only, no robot connection. Use it to check colours. | Any camera |

Why not just the webcam with URSim? The simulated robot can't move a real webcam, so the object
never re-centres and the robot drifts to the edge of the workspace. `--sim` and `--webcam-table`
close that loop.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--sim` | off | Simulated tool camera (needs URSim). |
| `--webcam-table` | off | PC webcam on the virtual table (needs URSim). |
| `--no-robot` | off | Don't connect to the robot. Not allowed with `--sim`, `--webcam-table` or `--calibrate`. |
| `--camera` | `0` | Camera index or device path. |
| `--color` | `red` | `red`, `yellow`, `green` or `blue`. |
| `--ip` | `127.0.0.1` | Robot IP. Any IP other than `127.0.0.1`/`localhost` requires `--start-here`. |
| `--gain` | `3.0` | Proportional gain (1/s): velocity = gain × offset. Lower it if the robot oscillates. |
| `--feedforward` | `1.0` | 0–1. Adds the object's estimated velocity, which removes most of the lag on a moving object. `0` = pure P control. |
| `--max-speed` | `0.15` | TCP speed cap in m/s, hard-limited to 0.25. |
| `--height` | `0.30` | TCP height (m) while following. Ignored with `--start-here`, which keeps the current height. |
| `--start-here` | off | Follow from the **current** pose (height and orientation). Never moves to a preset pose. Required on the real robot. |
| `--radius` | `0.15` | With `--start-here`: half-size (m) of the XY box around the start position. |
| `--calibrate` | off | Before following, move ±2 cm in X and Y and measure how a still object moves in the image (image Jacobian). Then the camera's mounting doesn't need to be known. |
| `--sim-mount YAW FLIP` | — | `--sim` only: actual camera mounting (yaw in degrees, mirror 0/1) to test `--calibrate`, e.g. `--sim-mount 90 1`. |
| `--headless` | off | No window (tests). |
| `--duration` | `0` | Stop after N seconds (`0` = run until `q`). |

Without `--start-here`, the script first does a `moveJ` to a standard elbow-up pose at
`START_XY`, `--height`, tool pointing down. That pose is meant for URSim only.

## Parameters in the script

Constants at the top of `follow_object.py`:

| Constant | Value | Meaning |
|---|---|---|
| `ROBOT_IP` | `'127.0.0.1'` | Default for `--ip`. |
| `TOOL_DOWN` | `[0, π, 0]` | Tool orientation (rotation vector) in URSim mode: tool Z pointing down. |
| `WORKSPACE` | x `[-0.75, -0.20]`, y `[-0.45, 0.45]` m | Box the target is clamped to (URSim mode). With `--start-here` it becomes start ± `--radius`. |
| `FOLLOW_Z` | `0.30` m | Default for `--height`. |
| `START_XY` | `(-0.45, 0.0)` m | Start position in URSim mode. |
| `HOME_SEED_DEG` | `[-162.8, -88.1, -120.5, -61.4, 90.0, -72.8]` | Joint seed for the start pose's IK: elbow-up, every joint far from its limits. |
| `OBJECT_Z` | `0.0` m | Height of the object's top surface in the base frame. Used for depth when not calibrated. |
| `DT` | `0.02` s | Control period (50 Hz). |
| `GAIN` | `3.0` 1/s | Default for `--gain`. |
| `FEEDFORWARD` | `1.0` | Default for `--feedforward`. |
| `VEL_FILTER` | `0.15` | Low-pass on the object velocity estimate (0–1, lower = smoother). |
| `MAX_SPEED` | `0.15` m/s | Default for `--max-speed`. |
| `MAX_ACCEL` | `0.5` m/s² | Limit on velocity changes. |
| `DEADBAND_PX` | `4` px | Offsets smaller than this are ignored (detection jitter). |
| `ERROR_FILTER` | `0.5` | Low-pass on the measured offset (1 = no filter). |
| `LOST_TIMEOUT` | `0.5` s | Time without detection before the robot brakes to a stop. |
| `CAM_HFOV_DEG` | `60` ° | Horizontal field of view, for the focal length. Not needed with `--calibrate`. |
| `CAM_YAW_DEG` | `0` ° | Camera rotation about the tool Z axis. Not needed with `--calibrate`. |
| `CAM_FLIP` | `False` | Image mirrored relative to the tool frame. Not needed with `--calibrate`. |
| `COLOR_RANGES` | see code | HSV ranges per colour (OpenCV: H 0–179; red needs two ranges). |
| `MIN_AREA` | `300` px² | Smaller blobs are ignored. |
| `WebcamTable.ZOOM` | `1.8` | `--webcam-table`: size of the webcam image on the table relative to the tool camera's view. |

## Cameras

| Device | Camera |
|---|---|
| `/dev/video0` | Laptop webcam (1080p FHD) |
| `/dev/video6` | RealSense D435i, **colour** (use the stable `/dev/v4l/by-id/...-video-index0` path, video numbers can change when replugged) |
| `/dev/video4` | RealSense D435i, infrared (grey) |

**Colour choice with the RealSense:** the teach pendant's blue-green body triggers both `blue` and
`green`, and the cream table can trigger `yellow` and `green`. Use **red or blue** objects, keep the
pendant (and its red e-stop) out of view, and check with `--no-robot` that it prints `LOST` with no
object in view.

## Real robot

The RealSense is mounted on the tool, so the default mode closes the loop for real.

1. With the pendant, jog the robot so the camera looks down at the table from 25–40 cm. Move the
   pendant out of view.
2. Check detection without the robot:
   ```bash
   uv run python project/follow_object.py --no-robot --color red --camera "$RS"
   ```
3. Put the object under the camera and leave it still. Keep a hand on the e-stop. First run, slow
   and in a small box:
   ```bash
   uv run python project/follow_object.py --ip <ROBOT_IP> --start-here --calibrate \
       --max-speed 0.05 --gain 1.5 --radius 0.10 --color red --camera "$RS"
   ```
   `--calibrate` makes four slow 2 cm moves, prints the measured mapping (mm per pixel, image axis
   direction, mirrored or not) and refuses to start if the object left the view or moved.
4. When it follows smoothly, raise the speed and gain step by step, toward `--max-speed 0.15 --gain 3`.
   If it oscillates, lower `--gain`. Camera latency on the real robot is the main limit.

On an e-series robot, **Remote Control** must be on in Polyscope. If EtherNet/IP, PROFINET or MODBUS
is enabled on the robot, the connection fails with "input registers already in use".

## Test results (URSim)

| Test | Mean / max tracking error |
|---|---|
| `--sim`, default settings | 8–10 mm / 17 mm |
| P control only (`--feedforward 0`), gain 3 | 22 mm / 45 mm |
| `--sim-mount 90 1` without `--calibrate` | 291 mm (lost) |
| `--sim-mount 90 1 --calibrate` | 6 mm / 13 mm |
| `--sim-mount -135 0 --calibrate` | 9 mm / 18 mm |
| `--webcam-table`, fake webcam image, still object | 0.7 mm |

The simulated camera has no latency; expect a lower usable gain on the real robot.

## The cobot package

### Layout

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
| `project/follow_object.py` | **Object following** (this page) |
| `workshop/` | Workshop decks and notebooks, kept locally and git-ignored |

### Usage

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
