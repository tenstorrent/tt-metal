#!/bin/bash
# usage: mk_nz.sh: op_copy_nz/ = op_copy/ with every MaybeDeviceZoneScope line deleted (the unzoned A/B; select it with
# HRL_COPY_DIR=op_copy_nz). The hrl_ zones sit inside #ifdef HRL_ZONES, so deleting their lines changes nothing else.
E=$(cd "$(dirname "$0")" && pwd)
rm -rf $E/op_copy_nz && cp -r $E/op_copy $E/op_copy_nz && rm -rf $E/op_copy_nz/__pycache__
sed -i '/MaybeDeviceZoneScope(/d' $E/op_copy_nz/kernels/*.cpp $E/op_copy_nz/kernels/*.hpp
grep -c "MaybeDeviceZoneScope" $E/op_copy_nz/kernels/* | grep -v ":0" || echo "zones stripped"
