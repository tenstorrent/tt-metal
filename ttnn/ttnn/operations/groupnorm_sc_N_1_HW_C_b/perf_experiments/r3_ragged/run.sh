#!/usr/bin/env bash
# Profiled bench_ab.py run + parse. usage: R3_VARIANTS=... R3_CASES=... [R3_CHUNK_ZONES=1] run.sh <tag>
set -u
REPO=/localdev/mmoscicki/gn-regen/tt-metal
HERE=$REPO/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/r3_ragged
TAG=${1:-run}
mkdir -p $HERE/logs
LOG=$HERE/logs/$TAG.log
cd $REPO
source python_env/bin/activate
R3_RUNS=runs_$TAG.jsonl scripts/run_safe_pytest.sh --profile --no-py-zones --run-all $HERE/bench_ab.py --import-mode=prepend -s > $LOG 2>&1
echo "exit=$?  log=$LOG"
grep -E "SAFE_PYTEST_RESULT|SAFE_PYTEST: triage|AssertionError| passed| failed" $LOG | tail -4
CSV=$(grep -o "SAFE_PYTEST: PROFILER CSV: .*" $LOG | sed 's/SAFE_PYTEST: PROFILER CSV: //' | tail -1)
if [ -n "$CSV" ]; then
  cp $REPO/generated/profiler/.logs/profile_log_device.csv $HERE/logs/$TAG.device.csv 2>/dev/null
  cp "$CSV" $HERE/logs/$TAG.ops.csv
  python3 $HERE/../_round2/parse.py "$CSV" $HERE/runs_$TAG.jsonl | tee $HERE/logs/$TAG.txt
fi
