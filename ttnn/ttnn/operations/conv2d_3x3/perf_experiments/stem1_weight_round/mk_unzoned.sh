#!/usr/bin/env bash
# Regenerate the unzoned packages op_base_uz/ and op_copy_uz/ from op_base/ and op_copy/ (MaybeDeviceZoneScope lines
# stripped, as ablation_r11/ab_unzoned.sh does). Run after every kernel edit to op_copy/.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
for p in op_base op_copy op_grad; do
    rm -rf "$HERE/${p}_uz"
    mkdir -p "$HERE/${p}_uz"
    cp "$HERE/$p"/*.py "$HERE/${p}_uz/"
    cp -r "$HERE/$p/kernels" "$HERE/${p}_uz/"
    sed -i '/MaybeDeviceZoneScope(/d' "$HERE/${p}_uz"/kernels/*.cpp "$HERE/${p}_uz"/kernels/*.hpp
done
echo "unzoned: $(grep -c MaybeDeviceZoneScope "$HERE"/op_copy_uz/kernels/* | awk -F: '{s+=$2} END {print s}') zone lines left"
