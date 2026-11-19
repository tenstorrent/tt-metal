#!/bin/bash
# usage: mk_lz.sh : (re)build op_base_lz/ op_grad_lz/ op_copy_lz/ with every zone stripped except compute's kw-fold waits
# (compute_kwf_chunk_wait, compute_kwf_weight_wait, compute_kwf_halo_wait, compute_kwf_final, compute_kwf_side_rounds):
# a light-zoned build that keeps the data-movement RISCs' cold start-up paths unzoned.
cd "$(dirname "$0")"
for v in op_base op_copy op_grad; do
    rm -rf ${v}_lz && cp -r $v ${v}_lz
    sed -i '/MaybeDeviceZoneScope(/{/compute_kwf_chunk_wait\|compute_kwf_weight_wait\|compute_kwf_halo_wait\|compute_kwf_final\|compute_kwf_side_rounds/!d}' ${v}_lz/kernels/*.cpp ${v}_lz/kernels/*.hpp
done
# compute's slab waits (the subblock's in0 wait), zoned in the light builds only
for v in op_base op_copy op_grad; do
python3 - ${v}_lz/kernels/conv2d_3x3_kw_fold.hpp <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "                        if (pages > landed) {\n                            in.wait_front(pages);"
new = "                        if (pages > landed) {\n                            MaybeDeviceZoneScope(\"compute_kwf_slab_wait\");\n                            in.wait_front(pages);"
assert s.count(old) == 2, p
s = s.replace(old, new, 1)
open(p, "w").write(s)
PY
done
