#!/bin/bash
# usage: run_variant.sh <depth|auto> <case> [zones 0|1] [payload]
# one fresh profiled run (4 calls) of the op's perf harness with the depth patch; appends calls 2-4 max-over-chips to
# results.txt and keeps the device profile log (zones) as logs/<tag>.csv
set -e
ROOT=/localdev/mstaletovic/2026_10_06/1759_mstaletovic_mm_rs_eval/clones/matmul_reduce_scatter_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/matmul_reduce_scatter/perf_experiments/handoff_depth
cd $ROOT
depth=$1; case=$2; zones=${3:-1}; payload=${4:-}
tag="d${depth}_${case}_z${zones}${payload:+_p$payload}"
mkdir -p $D/logs
env PYTHONPATH=$D MMRS_HANDOFF_DEPTH=$depth MMRS_PERF_ZONES_SET=$zones $([ "$zones" = 1 ] && echo MMRS_PERF_ZONES=1) ${payload:+MMRS_PAYLOAD=$payload} \
  MMRS_CASES=$case MMRS_CALLS=4 scripts/run_safe_pytest.sh --profile \
  tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_perf.py -p depth_patch -s \
  > $D/logs/$tag.out 2>&1 || true
grep -E "handoff_depth\] policy|SAFE_PYTEST_RESULT" $D/logs/$tag.out | sort -u
csv=$(ls -td generated/profiler/reports/*/ | head -1)/ops_perf_results*.csv
cp generated/profiler/.logs/profile_log_device.csv $D/logs/$tag.csv 2>/dev/null || true
res=$(python3 /tmp/mmrs_perf_parse.py $csv | sed -n '3,5p' | awk '{print $4}' | paste -sd' ')
echo "$tag calls2-4 max_us: $res" | tee -a $D/results.txt
