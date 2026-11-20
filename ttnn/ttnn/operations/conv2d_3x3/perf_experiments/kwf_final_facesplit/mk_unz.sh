#!/bin/bash
# usage: mk_unz.sh : (re)build the zone-stripped twins op_base_unz/ and op_copy_unz/ from op_base/ and op_copy/.
E=/localdev/pmilojevic/workspaces/opgen/ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_final_facesplit
for v in base copy grad; do
    rm -rf $E/op_${v}_unz && cp -r $E/op_$v $E/op_${v}_unz && rm -rf $E/op_${v}_unz/__pycache__
    sed -i '/MaybeDeviceZoneScope(/d' $E/op_${v}_unz/kernels/*
done
