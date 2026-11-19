#!/usr/bin/env bash
# Zone-stripped twins of op_copy and grad (op_copy_nz, grad_nz), as perf_experiments/ablation_r10/ab_unzoned.sh strips
# the real op.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$HERE/op_copy_nz"
cp -r "$HERE/op_copy" "$HERE/op_copy_nz"
rm -rf "$HERE/op_copy_nz/__pycache__"
sed -i '/MaybeDeviceZoneScope(/d' "$HERE"/op_copy_nz/kernels/*.cpp "$HERE"/op_copy_nz/kernels/*.hpp
grep -c "MaybeDeviceZoneScope(" "$HERE"/op_copy_nz/kernels/* | grep -v ":0" || echo "op_copy_nz: zones stripped"
# The graduation copy (grad/ = HEAD + graduation.patch) likewise.
rm -rf "$HERE/grad_nz"
cp -r "$HERE/grad" "$HERE/grad_nz"
rm -rf "$HERE/grad_nz/__pycache__"
sed -i '/MaybeDeviceZoneScope(/d' "$HERE"/grad_nz/kernels/*.cpp "$HERE"/grad_nz/kernels/*.hpp
