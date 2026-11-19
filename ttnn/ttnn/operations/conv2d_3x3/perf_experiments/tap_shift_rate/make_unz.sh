#!/usr/bin/env bash
# Regenerate the unzoned copies (op_base_unz / op_copy_unz) from op_base / op_copy: MaybeDeviceZoneScope lines stripped.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
for c in op_base op_copy; do
    rm -rf "$HERE/${c}_unz"
    cp -r "$HERE/$c" "$HERE/${c}_unz"
    rm -rf "$HERE/${c}_unz/__pycache__"
    sed -i '/MaybeDeviceZoneScope(/d' "$HERE/${c}_unz"/kernels/*.cpp "$HERE/${c}_unz"/kernels/*.hpp
done
grep -c "MaybeDeviceZoneScope(" "$HERE"/op_*_unz/kernels/* | grep -v ":0" || true
