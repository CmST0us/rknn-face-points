#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
g++ -O2 -std=c++17 -fPIC -shared \
  "$root/src/face_tracker.cpp" "$root/src/gst_rknn_face.cpp" \
  -I"$root/include" -I"$root/src" -L"$root/lib" -lrknnrt \
  $(pkg-config --cflags --libs gstreamer-video-1.0) \
  -Wl,-rpath,/opt/face-rknn/lib \
  -o "$root/bin/libgstrknnface.so"
