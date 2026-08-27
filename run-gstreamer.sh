#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
input=$(readlink -f "${1:-$root/assets/test.jpg}")
[ -f "$input" ] || { echo "input not found: $input" >&2; exit 2; }
mkdir -p "$root/output"
cd "$root"
docker image inspect face-rknn-gstreamer:1.0.0 >/dev/null 2>&1 || docker compose build gstreamer-face
GST_INPUT_PATH="$input" docker compose run --rm --no-deps gstreamer-face
