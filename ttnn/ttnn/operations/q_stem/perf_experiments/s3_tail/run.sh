#!/bin/bash
# usage: run.sh <pkg> <mode> [knobs] [cases]   (from the repo root)
#   run.sh q_stem_v perf "S3I_MAX_ROWS=2"
#   run.sh q_stem_v zone ""       -> zones_<pkg>_<knobs>.csv + stage table
set -e
HERE=ttnn/ttnn/operations/q_stem/perf_experiments/s3_tail
PKG=$1; MODE=$2; KNOBS=${3:-}; CASES=${4:-0}
extra=()
[ "$MODE" = zone ] && export TT_METAL_DEVICE_PROFILER=1
QSTEM_PKG=$PKG QSTEM_MODE=$MODE QSTEM_KNOBS="$KNOBS" QSTEM_CASES=$CASES CCL_PERF_EXPERIMENTS=${CCL_PERF_EXPERIMENTS:-5} \
  CCL_PERF_OUT=$HERE/results timeout 2400 scripts/run_safe_pytest.sh --no-precompile --rootdir=$HERE $HERE/bench.py -s \
  > $HERE/last_run.log 2>&1 || true
grep -E "SAFE_PYTEST_RESULT|passed|failed|Error|error:|fastest|slowest|triage" $HERE/last_run.log | grep -v "^ccache" | tail -12
if [ "$MODE" = zone ]; then
  tag=$(echo "$KNOBS" | tr -c 'A-Za-z0-9_\n' '_')
  cp generated/profiler/.logs/profile_log_device.csv $HERE/zones_${PKG}_${tag}.csv
  python3 $HERE/tail_table.py $HERE/zones_${PKG}_${tag}.csv $HERE/results/coremap.json
fi
