#!/bin/bash
# Rebuild the *_nz copies: op_copy/, op_base/ (and grad/ when present) with every MaybeDeviceZoneScope line stripped.
E=$(cd "$(dirname "$0")" && pwd)
for d in op_copy op_base grad; do
    [ -d $E/$d ] || continue
    rm -rf $E/${d}_nz && cp -r $E/$d $E/${d}_nz && rm -rf $E/${d}_nz/__pycache__
    sed -i '/MaybeDeviceZoneScope(/d' $E/${d}_nz/kernels/*.cpp $E/${d}_nz/kernels/*.hpp
    echo "${d}_nz: $(cat $E/${d}_nz/kernels/* | grep -c MaybeDeviceZoneScope) zone lines left"
done
