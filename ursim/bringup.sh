#!/usr/bin/env bash
# Start URSim (if needed), power on the arm and release the brakes via the dashboard server.
# Remote Control mode must still be enabled once in Polyscope (http://localhost:6080/vnc.html).
set -euo pipefail
cd "$(dirname "$0")/.."

dash() { printf '%s\nquit\n' "$1" | timeout 5 nc 127.0.0.1 29999 | sed -n 2p; }
wait_for() { # $1 = command, $2 = pattern, $3 = timeout seconds
  local end=$((SECONDS + $3))
  until dash "$1" 2>/dev/null | grep -q "$2"; do
    (( SECONDS < end )) || { echo "timeout waiting for '$2' from '$1'" >&2; exit 1; }
    sleep 2
  done
}

docker compose up -d
echo "waiting for dashboard server..."; wait_for robotmode Robotmode 180
dash "close popup" >/dev/null
if ! dash robotmode | grep -q RUNNING; then
  dash "power on";      wait_for robotmode 'IDLE\|RUNNING' 60
  dash "brake release"; wait_for robotmode RUNNING 60
fi
echo "URSim ready: $(dash robotmode)  |  GUI: http://localhost:6080/vnc.html"
