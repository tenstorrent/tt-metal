#!/bin/bash
# usage: prof.sh <variant> <K>   (run from tt-metal root, through on-node)
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/launch_gap
OUT=$D/prof/$1_K$2
mkdir -p $OUT
LG_VARIANTS=$1 LG_K=$2 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=$OUT scripts/run_safe_pytest.sh --no-precompile $D/test_lg_profile.py 2>&1 | grep -E "PASS|FAIL|rror" | tail -3
