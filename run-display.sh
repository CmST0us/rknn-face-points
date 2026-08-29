#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source=${1:-/dev/video0}
runtime=${WAYLAND_RUNTIME_DIR:-/run/user/1000}
display=${WAYLAND_DISPLAY:-wayland-0}
drm=${DRM_DEVICE:-/dev/dri/card0}
[ -S "$runtime/$display" ] || { echo "Wayland socket not found: $runtime/$display" >&2; exit 2; }
cd "$root"
docker image inspect face-rknn-gstreamer:1.0.0 >/dev/null 2>&1 || docker compose build gstreamer-face

if [ -c "$source" ]; then
  exec docker compose run --rm --no-deps \
    -e "XDG_RUNTIME_DIR=$runtime" -e "WAYLAND_DISPLAY=$display" -v "$runtime:$runtime" \
    gstreamer-face -e v4l2src "device=$source" io-mode=dmabuf \
    "!" video/x-raw,format=NV12 \
    "!" queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream \
    "!" rknnfacemesh \
    "!" queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream \
    "!" waylandsink "drm-device=$drm" sync=false
fi

case "$source" in
  http://*|https://*)
    exec docker compose run --rm --no-deps \
      -e "XDG_RUNTIME_DIR=$runtime" -e "WAYLAND_DISPLAY=$display" -v "$runtime:$runtime" \
      gstreamer-face -e souphttpsrc "location=$source" "!" hlsdemux "!" parsebin name=p \
      p. "!" video/x-h264 "!" queue "!" h264parse \
      "!" mppvideodec dma-feature=true format=NV12 "!" queue \
      "!" rknnfacemesh \
      "!" queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream \
      "!" waylandsink "drm-device=$drm" sync=true
    ;;
esac

input=$(readlink -f "$source")
[ -f "$input" ] || { echo "input not found: $input" >&2; exit 2; }
GST_INPUT_PATH="$input" exec docker compose run --rm --no-deps \
  -e "XDG_RUNTIME_DIR=$runtime" -e "WAYLAND_DISPLAY=$display" -v "$runtime:$runtime" \
  gstreamer-face -e filesrc location=/input.jpg "!" qtdemux name=d \
  d.video_0 "!" queue "!" h264parse "!" mppvideodec dma-feature=true format=NV12 \
  "!" rknnfacemesh "!" waylandsink "drm-device=$drm"
