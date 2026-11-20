#!/usr/bin/env bash
# Regenerate the unzoned packages <pkg>_uz/ from <pkg>/ (MaybeDeviceZoneScope lines stripped). Run after every kernel
# edit to op_copy/.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
for p in ${@:-op_base op_copy}; do
    rm -rf "$HERE/${p}_uz"
    mkdir -p "$HERE/${p}_uz"
    cp "$HERE/$p"/*.py "$HERE/${p}_uz/"
    cp -r "$HERE/$p/kernels" "$HERE/${p}_uz/"
    sed -i "/R0DIAG-BEGIN/,/R0DIAG-END/d" "$HERE/${p}_uz"/kernels/*.cpp "$HERE/${p}_uz"/kernels/*.hpp
    sed -i '/MaybeDeviceZoneScope(/d' "$HERE/${p}_uz"/kernels/*.cpp "$HERE/${p}_uz"/kernels/*.hpp
done
echo "unzoned: $(grep -c MaybeDeviceZoneScope "$HERE"/op_copy_uz/kernels/* | awk -F: '{s+=$2} END {print s}') zone lines left"
