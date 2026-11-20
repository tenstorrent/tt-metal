#!/bin/bash
# usage: mk_unz.sh <name>: op_<name> -> op_<name>_unz with every MaybeDeviceZoneScope line stripped
E=/localdev/pmilojevic/workspaces/opgen/ttnn/ttnn/operations/conv2d_3x3/perf_experiments/s2_weight_slots
rm -rf $E/op_$1_unz && cp -r $E/op_$1 $E/op_$1_unz && rm -rf $E/op_$1_unz/__pycache__
sed -i '/MaybeDeviceZoneScope(/d' $E/op_$1_unz/kernels/*
grep -c MaybeDeviceZoneScope $E/op_$1_unz/kernels/* | grep -v ":0" || true
