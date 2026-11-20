#!/bin/bash
# usage: mk_unz.sh : (re)build op_*_unz/ from op_base/, op_copy/, op_grad/ with every zone stripped.
cd "$(dirname "$0")"
for v in op_base op_copy op_grad; do
    [ -d $v ] || continue
    rm -rf ${v}_unz && cp -r $v ${v}_unz
    sed -i '/MaybeDeviceZoneScope(/d' ${v}_unz/kernels/*.cpp ${v}_unz/kernels/*.hpp
done
