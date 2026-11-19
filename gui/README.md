# follow_launcher (Qt6)

Single-window Qt6 C++ app that configures and launches
`project/follow_object.py` via `uv run`. It does **not** control the robot or
OpenCV windows itself: those stay in the Python process. The GUI only builds
the CLI, shows logs, and can stop the process.

## Requirements

- CMake ≥ 3.16
- Qt6 Widgets (`qt6-base-dev` on Debian/Ubuntu, `qt` / `qt@6` on Homebrew)
- `uv` on `PATH` (or set the path in **Avancé**)

## Build

From the repository root:

```bash
# macOS (Homebrew)
cmake -S gui -B gui/build -DCMAKE_PREFIX_PATH="$(brew --prefix qt6)"
cmake --build gui/build

# Linux (system Qt6)
cmake -S gui -B gui/build
cmake --build gui/build
```

## Run

```bash
./gui/build/follow_launcher
```

Working directory for the Python script is the repo root (auto-detected by
walking up from the executable until `project/follow_object.py` is found).
Override it under **Avancé** if needed.

## Scenarios

| Scenario | What it adds |
|---|---|
| Détection seule | `--no-robot` |
| Robot lab | `--ip`, `--home` or `--start-here`, calibration mode |
| URSim | `--sim` or `--webcam-table` |

Target: colour only, or colour + `--shape` (optional photo path). Useful
controls: camera, gain, feedforward, max speed, radius, follow-depth.
Everything else is under **Avancé**.

**Lancer** starts `uv run python project/follow_object.py …`. **Arrêter**
sends `SIGTERM`, then `SIGKILL` after 2 s if still running.
