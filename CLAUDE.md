# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Teaching material for a workshop called "IA et Cobotique" (AI and collaborative robotics), held at Telecom Nancy. It is not a software project: there is no build system, no test suite and no git history. The workshop covers controlling a Universal Robots **UR5** arm through the **URSim** simulator from Python.

Each day has a French lecture deck `AI-and-Cobotics-Workshop-JourN_with_Gamma.pptx` and an English student notebook `DayN_*_Student.ipynb`, plus an optional instructor `DayN_*_Solution.ipynb`. The decks are 15–23 MB, so extract their text with `zipfile` (the `ppt/slides/slideN.xml` entries) rather than reading them directly.
- **Day 1, URSim setup** (16 Nov): AI paradigms, the perception → planning → action loop, UR5 kinematics, URSim, RTDE/URScript. The notebook covers connecting with `ur_rtde`, `moveJ`, `safe_move`, and logging the TCP pose to `trajectory.csv` (Polars + matplotlib).
- **Day 2, computer vision** (17 Nov): the OpenCV HSV colour pipeline, contours and centroids, checkerboard camera–robot calibration, MobileNet-SSD detection, `pixel_to_robot()`, and a vision-guided pick challenge in URSim. It uses webcam `/dev/video0`. Calibration uses the robot coordinates of three measured board corners (A, B, C), and `find_board()` re-orders the detected corners so that A is always the image's top-left corner. Don't simplify this to origin + angle: the detector can start from either end of the board, and a downward-looking camera mirrors the table, so a wrong orientation still fits with near-zero reprojection error but sends the robot 40 cm off. `Day2_ComputerVision_Student.original.ipynb` is the user's pre-rewrite version.
- **Day 3** (announced in the Day 2 deck): path planning with A* and RRT, decision logic, an autonomous pick-and-place workflow.

## Conventions

- **Student notebooks contain deliberate `# Student Task` / `TODO` stubs.** Exercise 4 and the Challenge are meant to be left for students to complete. Don't fill them in unless asked; put a solution in a separate notebook (e.g. `*_Solution.ipynb`).
- Keep the languages as they are: slides in French, notebooks in English, unless asked otherwise.
- Edit `.ipynb` files with the NotebookEdit tool, not as raw JSON.

## Running the notebooks

The robot notebooks need a running URSim instance (standalone or in a VirtualBox VM with a host-only adapter), set to **Remote Control** mode in Polyscope. The default target is `ROBOT_IP = '127.0.0.1'`.

The environment is managed with **uv** (`pyproject.toml`, `uv.lock`, `.venv/`). Python is pinned to **3.12** (`requires-python = ">=3.12,<3.13"`) because `ur_rtde` 1.6.5 only publishes wheels up to cp312. A newer Python would need a source build against Boost.

```bash
uv sync                                   # create/refresh .venv
uv run jupyter lab                        # or select the "Python (cobotique)" kernel in VS Code
uv add <pkg>                              # add a dependency
uv run jupyter nbconvert --to notebook --execute Day1_URSim_Setup_Student.ipynb --output /tmp/out.ipynb   # headless run
```

`ur_rtde` provides the `rtde_control` and `rtde_receive` modules. **OpenCV is pinned `<5`** because OpenCV 5 removed `cv2.dnn.readNetFromCaffe`, which Day 2 Exercise 3 uses. The `MobileNetSSD_deploy.{prototxt,caffemodel}` files in the project root are chuanqi305/MobileNet-SSD's `deploy.prototxt` and `mobilenet_iter_73000.caffemodel`, renamed. Day 2 uses `cv2.imshow` windows (Qt5 build, runs through XWayland), so those cells need a desktop session and block until a key is pressed in the window. A Jupyter kernel named `cobotique` is registered in the user's kernelspecs and points at `.venv`.

### URSim (Docker)

URSim runs from `compose.yaml` using the official `universalrobots/ursim_e-series` image with `ROBOT_MODEL=UR5`. That is Polyscope 5.x, so the robot is really a UR5e. The e-series was chosen because the notebook relies on Remote Control mode, which only exists there. Ports are bound to localhost only; `ursim/programs` and `ursim/urcaps` are mounted volumes.

```bash
ursim/bringup.sh        # docker compose up -d + power on + brake release (idempotent)
docker compose down     # stop
```

- GUI: http://localhost:6080/vnc.html. **Remote Control** must be on, or `RTDEControlInterface` fails with `Failed to start RTDE data synchronization`. The "enabled" setting lives in `/ursim/.polyscope/remotecontrol.properties`, which is bind-mounted read-only from `ursim/polyscope/`. With it, a freshly created container accepts `RTDEControlInterface` right after `bringup.sh`, with no clicks in the GUI (verified 2026-11-16). Polyscope settings are in `/ursim/.polyscope/`; controller config and joint limits are in `/ursim/.urcontrol/` (`urcontrol.conf`, `safety.conf` → `*.UR5`).
- On URSim, the dashboard's `is in remote control` reply reads `false` even when remote control works, so don't use it as a check.
- Only **one** `RTDEControlInterface` can be connected at a time. A second one fails with `One of the RTDE input registers are already in use!`, which usually means a notebook kernel still holds the connection; call `rtde_c.stopScript()` or restart the kernel. Any number of `RTDEReceiveInterface` clients can connect.

- **`ur_rtde` 1.6.5 + Polyscope 5.26: `rtde_c.getForwardKinematics(q)` returns wrong poses for an explicit `q`.** Called with no arguments, it is correct (current pose). The solution notebook computes FK itself from the UR5e DH parameters instead; this matches `getActualTCPPose()` to within 1e-10 m.

UR ports the material refers to: 30001–30004 for URScript and dashboard traffic, 30002 for the secondary client interface, 30004 for RTDE. The code cells cannot run without a reachable URSim: the `RTDEControlInterface` constructor fails immediately if none is found.
