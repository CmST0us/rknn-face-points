#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
device=${1:-/dev/video0}
export V4L2_DEVICE="$device"
mkdir -p "$root/output"
cd "$root"
docker image inspect face-rknn-gstreamer:1.0.0 >/dev/null 2>&1 || docker compose build gstreamer-face
exec docker compose run --rm --no-deps gstreamer-face \
  -e v4l2src "device=$device" \
  "!" videoconvert \
  "!" video/x-raw,format=RGB \
  "!" rknnfacemesh \
  "!" videoconvert \
  "!" jpegenc \
  "!" matroskamux \
  "!" filesink location=/output/face-output.mkv
