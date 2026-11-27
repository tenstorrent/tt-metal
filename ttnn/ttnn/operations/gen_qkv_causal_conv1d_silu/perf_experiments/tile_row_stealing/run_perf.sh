#!/bin/bash
# Usage: run_perf.sh <variant>-<case> [...]   e.g. run_perf.sh base-conv1d-004 steal-conv1d-004
# One op call per id, in the order given, ONE session (no --profile: the device profiler is enabled through
# the environment and read in-process by test_perf, which prints "STEAL_NS <id> <device kernel ns>").
# Profiler artifacts go to a private dir (the sibling optimizer shares generated/profiler); the device log of
# this invocation (stage zones, TS_DATA steal_blocks, one run host ID per op, in order) is kept as
# $PROF/saved/<n>.csv (path printed).
ROOT=/localdev/nbabin/2026_11_26/1733_nbabin_gen_qkv_causal_conv1d_silu/clones/gen_qkv_causal_conv1d_silu_run1/tt-metal
F=ttnn/ttnn/operations/gen_qkv_causal_conv1d_silu/perf_experiments/tile_row_stealing/test_steal.py
export TT_METAL_PROFILER_DIR=/tmp/tile_row_steal_prof
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
mkdir -p $TT_METAL_PROFILER_DIR/saved
rm -f $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv
cd "$ROOT"
ids=()
for v in "$@"; do ids+=("$F::test_perf[$v]"); done
GQKV_STEAL_PERF=1 timeout 560 scripts/run_safe_pytest.sh --run-all --precompile "${ids[@]}" > /tmp/steal_perf_last.log 2>&1
grep -E "SAFE_PYTEST_RESULT|[0-9]+ (passed|failed)" /tmp/steal_perf_last.log | tail -2
n=$(ls $TT_METAL_PROFILER_DIR/saved/*.csv 2>/dev/null | wc -l)
saved=$TT_METAL_PROFILER_DIR/saved/$n.csv
cp $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv $saved 2>/dev/null && echo "DEVICE_LOG $saved"
echo "$n $*" >> $TT_METAL_PROFILER_DIR/saved/index.txt
grep -o "STEAL_NS .*" /tmp/steal_perf_last.log | awk 'NF >= 3 {printf "%-40s %10s ns\n", $2, $3}'
