#!/usr/bin/env bash
# Fetch the MobileNet-SSD weights used by Day 2 Exercise 3 (chuanqi305/MobileNet-SSD,
# mobilenet_iter_73000.caffemodel) and verify the checksum.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() { # $1 = url, $2 = target, $3 = sha256
  if [[ -f $2 ]] && echo "$3  $2" | sha256sum -c --quiet 2>/dev/null; then
    echo "$2 already present"; return
  fi
  echo "downloading $2"
  curl -fsSL -o "$2.part" "$1"
  echo "$3  $2.part" | sha256sum -c --quiet
  mv "$2.part" "$2"
}

BASE=https://github.com/chuanqi305/MobileNet-SSD/raw/master
fetch "$BASE/mobilenet_iter_73000.caffemodel" MobileNetSSD_deploy.caffemodel \
  52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7
fetch "$BASE/deploy.prototxt" MobileNetSSD_deploy.prototxt \
  2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608
