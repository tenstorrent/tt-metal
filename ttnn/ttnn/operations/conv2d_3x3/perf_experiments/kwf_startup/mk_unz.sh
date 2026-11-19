#!/bin/bash
# usage: mk_unz.sh : (re)build op_base_unz/ and op_copy_unz/ from op_base/ and op_copy/ with every zone stripped.
cd "$(dirname "$0")"
for v in op_base op_copy op_grad; do
    rm -rf ${v}_unz && cp -r $v ${v}_unz
    sed -i '/MaybeDeviceZoneScope(/d' ${v}_unz/kernels/*.cpp ${v}_unz/kernels/*.hpp
    grep -c "MaybeDeviceZoneScope(" ${v}_unz/kernels/* | grep -v ":0" || true
done
