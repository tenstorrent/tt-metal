#!/bin/bash
# Regenerates op_copy/kernels_nozone/ from op_copy/kernels/ with every MaybeDeviceZoneScope line removed, for whole-op
# numbers without the reader zones' profiled overhead (see the conv2d_3x3 lessons: zones on a DM RISC shift profiled
# A/B numbers). Every MaybeDeviceZoneScope sits alone on its line.
set -euo pipefail
cd "$(dirname "$0")/op_copy"
rm -rf kernels_nozone
cp -r kernels kernels_nozone
sed -i '/MaybeDeviceZoneScope(/d' kernels_nozone/*.cpp kernels_nozone/*.hpp
grep -c "MaybeDeviceZoneScope(" kernels_nozone/* | grep -v ":0" || true
