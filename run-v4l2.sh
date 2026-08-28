#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
device=${1:-/dev/video0}
mkdir -p "$root/output"
cd "$root"
docker image inspect face-rknn-gstreamer:1.0.0 >/dev/null 2>&1 || docker compose build gstreamer-face
exec docker compose run --rm --no-deps gstreamer-face \
  -e v4l2src "device=$device" io-mode=dmabuf \
  "!" video/x-raw,format=NV12 \
  "!" queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 \
  "!" rknnfacemesh \
  "!" queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 \
  "!" mpph264enc bps=12000000 gop=60 header-mode=each-idr \
  "!" h264parse \
  "!" matroskamux \
  "!" filesink location=/output/face-output.mkv
