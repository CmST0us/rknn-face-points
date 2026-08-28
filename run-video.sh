#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
input=$(readlink -f "${1:-$root/assets/test.mov}")
[ -f "$input" ] || { echo "input not found: $input" >&2; exit 2; }
mkdir -p "$root/output"
cd "$root"
docker image inspect face-rknn-gstreamer:1.0.0 >/dev/null 2>&1 || docker compose build gstreamer-face
GST_INPUT_PATH="$input" docker compose run --rm --no-deps gstreamer-face -e \
  filesrc location=/input.jpg ! qtdemux name=d \
  d.video_0 ! queue ! h264parse ! mppvideodec dma-feature=true format=NV12 ! \
  queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! rknnfacemesh ! \
  queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  mpph264enc bps=12000000 gop=60 header-mode=each-idr ! \
  h264parse ! mp4mux ! filesink location=/output/face-output.mp4
