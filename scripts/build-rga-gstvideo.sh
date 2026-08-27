#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
docker buildx build --platform linux/arm64 --target export \
  --output "type=local,dest=$out" -f "$root/Dockerfile.gstvideo-builder" "$root"
cp "$out/libgstvideo-1.0.so.0.2402.0" "$root/rockchip/"
