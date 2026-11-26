#!/usr/bin/env bash
# Profiled bench.py run + parse. usage: CT_VARIANTS=... CT_CASES=... run.sh <tag> [--dev]
set -u
REPO=/localdev/mmoscicki/gn-regen/tt-metal
HERE=$REPO/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/combine_tail
TAG=${1:-run}
MODE=${2:-}
mkdir -p $HERE/logs
LOG=$HERE/logs/$TAG.log
export CT_RUNS=runs_$TAG.jsonl
cd $REPO
source python_env/bin/activate
if [ "$MODE" = "--dev" ]; then
  scripts/run_safe_pytest.sh --dev --run-all $HERE/bench.py --import-mode=prepend -s > $LOG 2>&1
else
  scripts/run_safe_pytest.sh --profile --no-py-zones --run-all $HERE/bench.py --import-mode=prepend -s > $LOG 2>&1
fi
echo "exit=$?  log=$LOG"
grep -E "SAFE_PYTEST_RESULT|SAFE_PYTEST: triage|AssertionError| passed| failed" $LOG | tail -4
CSV=$(grep -o "SAFE_PYTEST: PROFILER CSV: .*" $LOG | sed 's/SAFE_PYTEST: PROFILER CSV: //' | tail -1)
if [ -n "$CSV" ]; then
  cp $(dirname $CSV)/../../.logs/profile_log_device.csv $HERE/logs/$TAG.device.csv 2>/dev/null || cp $REPO/generated/profiler/.logs/profile_log_device.csv $HERE/logs/$TAG.device.csv
  python3 $HERE/parse.py "$CSV" $HERE/$CT_RUNS --save $TAG | tee $HERE/logs/$TAG.txt
fi
python3 - <<PY
import json
for l in open("$HERE/$CT_RUNS"):
    r = json.loads(l); print(f"{r['case']:28s} {r['variant']:14s} cpi {r.get('cpi','-')} blk {r.get('max_blocks','-')} pcc {r['pcc']:.6f} rms {r['rms']:.5f} ndiff {r.get('ndiff','-')} maxdiff {r.get('maxdiff','-')}")
PY
