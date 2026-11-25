#!/bin/bash
# usage (repo root): run.sh <pkg> <mode perf|zone|check> [knobs] [cases] [tag]
#   zone: QSTEM_BLKZ=1 (per-block port zones), per-port table -> results/ports_<pkg>_<tag>.txt
HERE=ttnn/ttnn/operations/q_stem/perf_experiments/send_schedule
PKG=$1; MODE=$2; KNOBS=${3:-}; CASES=${4:-0}; TAG=${5:-$(echo "$KNOBS" | tr -c 'A-Za-z0-9_\n' '_')}
[ "$MODE" = zone ] && export TT_METAL_DEVICE_PROFILER=1 QSTEM_BLKZ=${QSTEM_BLKZ:-1}
QSTEM_PKG=$PKG QSTEM_MODE=$MODE QSTEM_KNOBS="$KNOBS" QSTEM_CASES=$CASES CCL_PERF_EXPERIMENTS=${CCL_PERF_EXPERIMENTS:-5} \
  CCL_PERF_OUT=${CCL_PERF_OUT:-$HERE/results} timeout 2400 scripts/run_safe_pytest.sh --no-precompile --rootdir=$HERE \
  $HERE/bench.py${TEST:-::test_bench} -s > $HERE/last_run.log 2>&1
echo "rc=$?"
grep -E "SAFE_PYTEST_RESULT|SEND_SCHED_METRICS|passed|failed|Error|error:|fastest|slowest|triage" $HERE/last_run.log | grep -v "^ccache" | tail -14
if [ "$MODE" = zone ]; then
  cp generated/profiler/.logs/profile_log_device.csv $HERE/results/zones_${PKG}_${TAG}.csv
  python3 $HERE/ports.py $HERE/results/plan.json $HERE/results/zones_${PKG}_${TAG}.csv | tee $HERE/results/ports_${PKG}_${TAG}.txt
fi
