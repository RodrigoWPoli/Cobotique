# Cobotique: UR5e object following

### TLDR:


**Once per machine** (`data/`, where the home pose and calibration are saved, is not in git):

```bash
# 1. With the pendant, jog the robot to the start position, camera looking forward, image the right way up.
uv run python project/follow_object.py --ip 100.65.63.93 --set-home      # saves it, no motion

# 2. Calibrate from home and start following (moves the robot: 4 slow 5 cm moves, then follows).
#    Hold the blue object ~50 cm in front of the camera; it may move during calibration.
uv run python project/follow_object.py --ip 100.65.63.93 --home --calibrate-scene \
    --max-speed 0.01 --gain 1.5 --color blue --camera realsense --shape
```

**Every time after that:**

```bash
uv run python project/follow_object.py --ip 100.65.63.93 --home --load-calib \
    --max-speed 0.01 --gain 1.5 --color blue --camera realsense --shape
```

If `--home` refuses ("jog the robot closer to it first"), jog the robot near the start position and run it
again. Press `q` in the window to stop. `--shape` learns the silhouette from the newest photo in
`data/` (the blue bin) and ignores other blue blobs. Drop `--shape` to follow any blob of that colour.

**The UR5e follows a coloured object seen by a camera mounted on its tool** (eye-in-hand, image-based
visual servoing). Each frame, `project/follow_object.py`:

1. detects the object by colour (HSV, the same pipeline as Day 2). With `--shape`, only a blob whose silhouette matches a reference photo is kept,
2. measures the object's offset from the image centre,
3. turns that offset into a TCP velocity in the robot base frame: proportional control plus
   feedforward of the object's estimated velocity, with deadband, speed and acceleration limits,
4. integrates the velocity into a target pose and streams it with `servoL` at 50 Hz.

The TCP keeps its orientation and only moves in the plane at right angles to the camera's view: base
X/Y for a camera looking down, or, with `--start-here`, the start pose's tool X/Y. A camera looking
**forward** then follows sideways and up/down, and never moves toward the object. The target is always
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

## GUI launcher (Qt6)

Optional single-window Qt6 app that picks a scenario (detection / lab robot / URSim),
builds the CLI, shows logs, and can stop the process: see [`gui/README.md`](gui/README.md).

```bash
cmake -S gui -B gui/build -DCMAKE_PREFIX_PATH="$(brew --prefix qt6)"   # macOS
cmake --build gui/build && ./gui/build/follow_launcher
```

## Commands

All commands run from the repository root.

| Goal | Command |
|---|---|
| Simulated tool camera in URSim (closed loop) | `uv run python project/follow_object.py --sim` |
| Self-test, prints the tracking error | `uv run python project/follow_object.py --sim --headless --duration 25` |
| PC webcam with URSim | `uv run python project/follow_object.py --webcam-table --color red` |
| Check detection only, no robot | `uv run python project/follow_object.py --no-robot --color blue --camera realsense` |
| **Real robot** | see [Real robot](#real-robot) |

`--camera realsense` opens colour + aligned depth via librealsense (see [Cameras](#cameras)).

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
| `--camera` | `0` | Camera index, device path, or `realsense` (librealsense colour + aligned depth; see [Cameras](#cameras)). |
| `--color` | `red` | `red`, `yellow`, `green` or `blue`. |
| `--shape [PHOTO]` | off | Also match the object's **silhouette**, learned from `PHOTO`. With no path, the newest image in `data/` is used. The object must be the largest blob of `--color` in that photo. Live, a colour blob is kept only if its outline stays close to that silhouette (aspect ratio, how filled it is, Hu moments). Size and in-plane rotation do not matter, so a circle, a long bar, or an irregular blue region is ignored. The window draws the matched contour in green and prints `shape 0.xx` (0 = identical, kept up to 1.0). |
| `--ip` | `127.0.0.1` | Robot IP. Any IP other than `127.0.0.1`/`localhost` requires `--start-here`. |
| `--gain` | `1.0` | Proportional gain (1/s): velocity = gain × offset. Lower it if the robot oscillates. |
| `--feedforward` | `0.3` | 0–1. Adds the object's estimated velocity, which removes most of the lag on a moving object. `0` = pure P control. |
| `--max-speed` | `0.02` | TCP speed cap in m/s (2 cm/s), also used for the calibration moves. Hard-limited to 0.08 for safety. Use `0.01` for 1 cm/s. |
| `--height` | `0.30` | TCP height (m) while following. Ignored with `--start-here`, which keeps the current height. |
| `--start-here` | off | Follow from the **current** pose and orientation, moving in the plane of the start pose's tool X/Y (at right angles to the camera). Never moves to a preset pose. Required on the real robot. If the camera doesn't look down, `--calibrate` is required too. |
| `--set-home` | off | Save the current TCP position as the home pose, with the orientation **squared to the base axes** (tool Z and X snapped to the nearest ±X/Y/Z), then exit. Checks the pose is reachable; **does not move** the robot. |
| `--home` | off | Move slowly (`moveJ`, 0.25 rad/s) to the saved home pose, then follow from there (implies `--start-here`). Refuses if any joint would turn more than 45°: jog closer first. |
| `--home-file` | `data/follow_home.json` | Where `--set-home` saves and `--home` reads (git-ignored). |
| `--radius` | `0.25` | With `--start-here`: how far (m) the target may go from the start position along each plane axis. At start-up the box is **reduced automatically** where the arm can't reach (no IK solution, outside the safety limits, or a joint would turn more than 90°) or where the TCP would go below `--min-z`; the allowed region is printed. |
| `--follow-depth` | off | Also move along the **camera axis** to keep the object at `--target-distance`, using the depth measured at the object. Needs a depth camera (`--camera realsense`) and `--start-here` or `--home`. Without valid depth (object lost, nothing in range) the robot does not move along that axis; a sudden depth jump (a hand in front, a depth hole) is ignored unless it lasts 5 frames. Deadband ±2 cm. The `--radius` box and reach check then cover all three axes, and `--max-speed` caps the combined 3D speed. |
| `--target-distance` | `0.5` | Camera–object distance (m) kept with `--follow-depth`. |
| `--depth-gain` | `1.0` | Gain (1/s) of the distance controller: velocity along the camera axis = gain × distance error. |
| `--min-z` | `0.10` | With `--start-here`: lowest TCP height (m, robot base frame), to stay clear of the table the robot stands on. Raise it if the tool is long or there are objects on the table. |
| `--calibrate` | off | Before following, move ±`--calib-step` along both plane axes and measure how a still object moves in the image (image Jacobian), then save it to `--calib-file`. The camera's mounting and the object's distance don't need to be known. Each position is the median of 20 frames; if the object drifts or the + and − shifts don't mirror each other, the object moved and calibration is refused. |
| `--calibrate-scene` | off | Calibrate from the motion of the **whole scene** (ORB feature matches between frames): no object needed, so you can hold it in your hand, or not show it at all. Gives the image directions (rotation, mirroring) exactly; the scale comes from the focal length and RealSense depth (or `--object-distance` as fallback). Needs a textured scene (not a blank wall) and nothing large moving in view. Saves like `--calibrate`. |
| `--object-distance` | `0.5` | Fallback camera–object distance (m) for `--calibrate-scene` when RealSense depth is unavailable. With `--camera realsense`, depth at the image centre is used instead when valid. |
| `--calib-step` | `0.05` | Size (m) of the calibration moves. Bigger = less sensitive to a slightly moving object. Keep it below `--radius`. |
| `--load-calib` | off | Use the calibration saved by the last `--calibrate` / `--calibrate-scene`: no calibration moves. It's stored relative to the **tool** (the camera is fixed to it), so it stays valid when the tool is turned or rolled. A warning is printed if the camera looks more than 30° away from where it was calibrated, since the object distance (scale) may then differ. |
| `--calib-file` | `data/follow_calibration.json` | Where `--calibrate` saves and `--load-calib` reads (`data/` is git-ignored). |
| `--sim-mount YAW FLIP` | — | `--sim` only: actual camera mounting (yaw in degrees, mirror 0/1) to test `--calibrate`, e.g. `--sim-mount 90 1`. |
| `--headless` | off | No window (tests). |
| `--duration` | `0` | Stop after N seconds (`0` = run until `q`). |

Without `--start-here`, the script first does a `moveJ` to a standard elbow-up pose at
`START_XY`, `--height`, tool pointing down. That pose is meant for URSim only.

## Parameters in the script

Constants at the top of `follow_object.py` (detection constants live in `project/detection.py`):

| Constant | Value | Meaning |
|---|---|---|
| `ROBOT_IP` | `'127.0.0.1'` | Default for `--ip`. |
| `TOOL_DOWN` | `[0, π, 0]` | Tool orientation (rotation vector) in URSim mode: tool Z pointing down. |
| `WORKSPACE` | x `[-0.75, -0.20]`, y `[-0.45, 0.45]` m | Box the target is clamped to (URSim mode). With `--start-here` it becomes start ± `--radius`, reduced to what the arm can reach. |
| `FOLLOW_Z` | `0.30` m | Default for `--height`. |
| `START_XY` | `(-0.45, 0.0)` m | Start position in URSim mode. |
| `HOME_SEED_DEG` | `[-162.8, -88.1, -120.5, -61.4, 90.0, -72.8]` | Joint seed for the start pose's IK: elbow-up, every joint far from its limits. |
| `OBJECT_Z` | `0.0` m | Height of the object's top surface in the base frame. Used for depth when not calibrated. |
| `DT` | `0.02` s | Control period (50 Hz). |
| `GAIN` | `1.0` 1/s | Default for `--gain`. |
| `FEEDFORWARD` | `0.3` | Default for `--feedforward`. |
| `VEL_FILTER` | `0.15` | Low-pass on the object velocity estimate (0–1, lower = smoother). |
| `MAX_SPEED` | `0.02` m/s | Default for `--max-speed`; also the speed of the calibration moves. |
| `MAX_ACCEL` | `0.15` m/s² | Limit on velocity changes. |
| `DEADBAND_PX` | `4` px | Offsets smaller than this are ignored (detection jitter). |
| `ERROR_FILTER` | `0.5` | Low-pass on the measured offset (1 = no filter). |
| `LOST_TIMEOUT` | `0.5` s | Time without detection before the robot brakes to a stop. |
| `CAM_HFOV_DEG` | `60` ° | Horizontal field of view, for the focal length. Not needed with `--calibrate`. |
| `CAM_YAW_DEG` | `0` ° | Camera rotation about the tool Z axis. Not needed with `--calibrate`. |
| `CAM_FLIP` | `False` | Image mirrored relative to the tool frame. Not needed with `--calibrate`. |
| `COLOR_RANGES` | see `project/detection.py` | HSV ranges per colour (OpenCV: H 0–179; red needs two ranges). |
| `MIN_AREA` | `300` px² | Smaller blobs are ignored (`project/detection.py`). |
| `SHAPE_MAX` | `1.0` | Max silhouette distance kept with `--shape` (`project/detection.py`). |
| `WebcamTable.ZOOM` | `1.8` | `--webcam-table`: size of the webcam image on the table relative to the tool camera's view. |

## Cameras

| Device | Camera |
|---|---|
| `/dev/video0` | Laptop webcam (1080p FHD); `--camera 0` (OpenCV) |
| `--camera realsense` | RealSense D435i via **librealsense** (`project/camera.py`): colour + depth aligned to colour |
| OpenCV index / path | Plain webcam path; on Linux the old V4L colour node still works if you pass it explicitly |

`--camera realsense` does **not** use OpenCV’s AVFoundation/V4L index (on macOS that often only
exposes the Depth device). It uses `pyrealsense2` (Linux/Windows) or `pyrealsense2-macosx` (macOS),
installed by `uv sync`. Depth (m) fills the scale for `--calibrate-scene` when valid, is stored in
`data/follow_calibration.json` as `distance_m`, and rescales the image Jacobian while following so
mm/px stays right as the object moves closer or farther. `--object-distance` is only the fallback
when depth is missing.

**macOS:** allow Camera for the terminal in System Settings → Privacy & Security → Camera. Prefer a
USB 3 port (no hub). If librealsense finds no device (`failed to set power state`), some macOS
versions need the script run with `sudo`. `ur-rtde` has no macOS wheels, so `uv sync` compiles it
(`brew install cmake boost` first).

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
   uv run python project/follow_object.py --no-robot --color red --camera realsense
   ```
3. Put the object under the camera and leave it still. Keep a hand on the e-stop. First run, slow
   and in a small box:
   ```bash
   uv run python project/follow_object.py --ip <ROBOT_IP> --start-here --calibrate \
       --max-speed 0.02 --gain 1.5 --color red --camera realsense
   ```
   `--calibrate` makes four slow 5 cm moves (at `--max-speed`, 2 cm/s by default), prints the measured mapping (mm per pixel, image axis
   direction, mirrored or not), refuses to start if the object left the view or moved, and saves
   the result. **Rest the object on something** during calibration rather than holding it: a hand
   wobbles by 1–2 cm, which is the size of what calibration measures.
   **Nowhere to rest it?** Use `--calibrate-scene` instead: it measures how the whole scene moves,
   so the object can be in your hand (or out of view). With `--camera realsense`, depth sets the
   scale; otherwise set `--object-distance` to roughly how far you'll hold the object.
   Next time, with the camera looking the same way, skip the moves with `--load-calib` instead of
   `--calibrate`.
4. When it follows smoothly, raise the speed and gain step by step, up to `--max-speed 0.08` (the hard cap) and `--gain 3`.
   If it oscillates, lower `--gain`. Camera latency on the real robot is the main limit.

On an e-series robot, **Remote Control** must be on in Polyscope. If EtherNet/IP, PROFINET or MODBUS
is enabled on the robot, the connection fails with "input registers already in use".

### Home pose

Jog the robot roughly where you want it, then save that spot with a clean orientation:

```bash
uv run python project/follow_object.py --ip 100.65.63.93 --set-home     # no motion
```

From then on `--home` moves there slowly at start-up and follows from it. The lab robot's home
is TCP (0.148, −0.749, 0.540) m, camera along −Y, image x along −X, image y along −Z (the camera
image is the right way up; the first home, with image x along +X, showed it upside down).

### Camera looking forward

With the tool horizontal, the robot follows an object held in front of the camera, moving sideways
and up/down within ±`--radius` (25 cm by default, less where the arm can't reach). Hold the object 30–80 cm from the camera; `--calibrate-scene`
calibrates from the room, so it doesn't matter if your hand moves. Make sure nothing else of that colour is in view (in the lab, the **red door** behind the robot
triggers `red`: use `blue`, or check first with `--no-robot`):

```bash
uv run python project/follow_object.py --ip 100.65.63.93 --start-here --calibrate-scene \
    --object-distance 0.5 --max-speed 0.02 --gain 1.5 --color blue --camera realsense
```

### Network

The lab robot is at **`100.65.63.93`** (UR5e, PolyScope 5.11); this PC's USB-Ethernet adapter uses
`100.65.63.101/16`. That subnet is inside Tailscale's `100.64.0.0/10` range, and Tailscale drops
replies from it that don't come through `tailscale0`: ARP works but ping and every UR port time out.
Run `sudo tailscale down` while using the robot (`sudo tailscale up` afterwards), or give the robot
and the adapter addresses outside `100.64.0.0/10`.

## Test results (URSim)

**Speed fix (18 Nov):** until then the control loop was paced with ur_rtde's `initPeriod()`/`waitPeriod()`,
which wait for the RTDE interface's own period (2 ms on the e-series), not for the script's 20 ms. The
loop ran at ~130 Hz while the target advanced as if it ran at 50 Hz, so the robot moved **2–3× faster
than `--max-speed`**. The loop now paces itself to 50 Hz and advances the target by the measured time step.
Peak TCP speed with `--max-speed 0.05` is now 5.4–5.8 cm/s (servo lag) instead of 12–13 cm/s. Results
below that were measured before the fix are flattered by the extra speed.

| `--follow-depth` test (forward camera, wall with a red disc, `--max-speed 0.05`) | Result |
|---|---|
| Wall at 0.62 m, target 0.5 m | approaches and holds 0.503–0.510 m |
| Wall at 0.40 m | backs off to 0.491 m (stops at the ±2 cm deadband) |
| Wall moving 0.55 ± 0.08 m (20 s period) | distance stays 0.48–0.52 m |
| Every 10th depth reading a bogus 0.2 m | ignored, holds 0.509 m |
| No depth at all | no motion along the camera axis (stays at 0.596 m) |

Measured with the earlier, faster defaults (`--gain 3 --feedforward 1 --max-speed 0.15`); the current
defaults are slower for safety, so the robot lags more behind a moving object.

| Test | Mean / max tracking error |
|---|---|
| `--sim`, default settings | 8–10 mm / 17 mm |
| P control only (`--feedforward 0`), gain 3 | 22 mm / 45 mm |
| `--sim-mount 90 1` without `--calibrate` | 291 mm (lost) |
| `--sim-mount 90 1 --calibrate` | 6 mm / 13 mm |
| `--sim-mount -135 0 --calibrate` | 9 mm / 18 mm |
| `--webcam-table`, fake webcam image, still object | 0.7 mm |
| Forward-looking camera (wall 50 cm ahead, image rotated 90°), `--start-here --calibrate` | 3.6 mm / 8.5 mm |
| Same, `--load-calib` (saved calibration, no moves) | 3.6 mm / 7.4 mm |
| Same, object wobbling ±5 mm during calibration | accepted, axis direction off by ~3°; 2.3 mm / 4.4 mm |
| Same, object wobbling ±3 cm during calibration | refused ("the object moved during calibration") |
| Textured wall 50 cm ahead, `--calibrate-scene`, object **moving** during calibration | 11.7 mm (object-based `--calibrate` in the same scene: 12.4 mm)¹ |
| Same, mirrored camera | 11.1 mm |
| Object 1.5 m away, `--object-distance 0.5` (scale 3× off) vs `1.5` | 33.9 mm vs 32.2 mm |

¹ The textured test scene is slower to render (~24 Hz loop instead of 50 Hz), hence the higher numbers than the plain wall.

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
| `project/detection.py` | HSV colour blobs + optional silhouette match (`--shape`) |
| `project/camera.py` | Webcam / RealSense (colour + aligned depth) |
| `gui/` | Qt6 launcher (`follow_launcher`) for scenarios + logs |
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
