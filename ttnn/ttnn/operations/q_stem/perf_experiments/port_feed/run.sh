#!/bin/bash
# usage (repo root): run.sh <pkg> <mode perf|zone|check> [knobs] [cases] [ablate] [tag]
#   zone: copies the profiler csv to results/zones_<tag>.csv and prints pf_summary
HERE=ttnn/ttnn/operations/q_stem/perf_experiments/port_feed
PKG=$1; MODE=$2; KNOBS=${3:-}; CASES=${4:-0}; ABL=${5:-}; TAG=${6:-${PKG}}
[ "$MODE" = zone ] && export TT_METAL_DEVICE_PROFILER=1
QSTEM_PKG=$PKG QSTEM_MODE=$MODE QSTEM_KNOBS="$KNOBS" QSTEM_CASES=$CASES QSTEM_ABLATE="$ABL" \
  CCL_PERF_EXPERIMENTS=${CCL_PERF_EXPERIMENTS:-5} CCL_PERF_OUT=${CCL_PERF_OUT:-$HERE/results} \
  timeout 2400 scripts/run_safe_pytest.sh --no-precompile --rootdir=$HERE $HERE/bench.py::test_bench -s \
  > $HERE/results/last_${TAG}.log 2>&1
grep -E "SAFE_PYTEST_RESULT|PORT_FEED_METRICS|passed|failed|Error|error:|fastest|slowest|triage" $HERE/results/last_${TAG}.log | grep -v "^ccache" | tail -14
if [ "$MODE" = zone ]; then
  cp generated/profiler/.logs/profile_log_device.csv $HERE/results/zones_${TAG}.csv
  python3 $HERE/pf_summary.py $HERE/results/zones_${TAG}.csv
fi
