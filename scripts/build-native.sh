#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
g++ -O2 -std=c++17 "$root/src/face_tracking_demo.cpp" \
  -I"$root/include" -I"$root/third_party" -L"$root/lib" \
  -Wl,-rpath,/opt/face-rknn/lib -lrknnrt \
  -o "$root/bin/face_tracking_demo"
