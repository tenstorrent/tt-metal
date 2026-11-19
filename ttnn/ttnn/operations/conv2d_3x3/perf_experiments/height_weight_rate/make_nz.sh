#!/bin/bash
# Rebuild op_copy_nz/ and op_head_nz/: op_copy/ and op_base/ with every MaybeDeviceZoneScope line stripped.
E=$(cd "$(dirname "$0")" && pwd)
for d in op_copy op_head grad; do
    rm -rf $E/${d}_nz && cp -r $E/$d $E/${d}_nz && rm -rf $E/${d}_nz/__pycache__
    sed -i '/MaybeDeviceZoneScope(/d' $E/${d}_nz/kernels/*.cpp $E/${d}_nz/kernels/*.hpp
    echo "${d}_nz: $(cat $E/${d}_nz/kernels/* | grep -c MaybeDeviceZoneScope) zone lines left"
done
