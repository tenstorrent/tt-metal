#!/bin/bash
# usage: guard_hash.sh <label>   sha256 of every Panoptic LOOSE output (asym_padding/panoptic_hashes.py)
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
L=$1; O=/tmp/perf19/guard; mkdir -p $O
PANOPTIC_HASH_OUT=$O/hash_$L.json timeout 570 scripts/tt-probe.sh conv2d_3x3 < ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/panoptic_hashes.py > $O/hash_$L.log 2>&1
grep -E "^CASES|TT_PROBE_RESULT|Error" $O/hash_$L.log | tail -2
