#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
input=$(readlink -f "${1:-$root/assets/test.jpg}")
[ -f "$input" ] || { echo "input not found: $input" >&2; exit 2; }
mkdir -p "$root/output"
cd "$root"
INPUT_PATH="$input" docker compose run --rm --build face-rknn
