#!/usr/bin/env bash
# Zone-stripped twins (op_base_nz, op_copy_nz, grad_nz) for the unzoned verdict numbers, as
# perf_experiments/ablation_r12/ab_unzoned.sh strips the real op.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
for v in op_base op_copy grad; do
    [[ -d "$HERE/$v" ]] || continue
    rm -rf "$HERE/${v}_nz"
    cp -r "$HERE/$v" "$HERE/${v}_nz"
    rm -rf "$HERE/${v}_nz/__pycache__"
    sed -i '/MaybeDeviceZoneScope(/d' "$HERE"/${v}_nz/kernels/*.cpp "$HERE"/${v}_nz/kernels/*.hpp
    grep -c "MaybeDeviceZoneScope(" "$HERE"/${v}_nz/kernels/* | grep -v ":0" || echo "${v}_nz: zones stripped"
done
