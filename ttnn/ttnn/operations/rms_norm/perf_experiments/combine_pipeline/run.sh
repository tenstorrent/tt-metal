#!/bin/bash
# run.sh <tag> <cells> <variants> [profile=1]: one bench session. Gives up (kills the still-WAITING wrapper) if the
# device lock is not acquired within ACQ_DEADLINE s, so a late acquisition never gets killed mid-run by the caller's
# timeout. Copies the profiler CSVs into runs/ in the same command (shared generated/profiler).
cd /localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
source python_env/bin/activate
D=ttnn/ttnn/operations/rms_norm/perf_experiments/combine_pipeline
T=$1; CELLS=$2; VARS=$3; PROF=${4:-1}
ACQ_DEADLINE=${ACQ_DEADLINE:-400}
rm -f $D/runs/${T}_labels.jsonl
LOG=$D/runs/$T.log
PF=""; [ "$PROF" = 1 ] && PF="--profile"
CP_CELLS=$CELLS CP_VARIANTS=$VARS CP_TAG=$T scripts/run_safe_pytest.sh $PF $D/bench_combine_pipeline.py -s --import-mode=prepend > $LOG 2>&1 &
PID=$!
t=0
while kill -0 $PID 2>/dev/null; do
  if ! grep -q "Device lock acquired" $LOG && [ $t -ge $ACQ_DEADLINE ]; then
    pkill -P $PID; kill $PID; wait $PID 2>/dev/null; echo "GAVE_UP waiting for device lock after ${t}s"; exit 9
  fi
  sleep 5; t=$((t+5))
done
wait $PID; rc=$?
echo "rc=$rc"
grep -E "SAFE_PYTEST_RESULT|passed|failed|triage report" $LOG | tail -4
if [ "$PROF" = 1 ]; then
  csv=$(sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p' $LOG)
  [ -n "$csv" ] && cp "$csv" $D/runs/${T}_ops.csv && cp generated/profiler/.logs/profile_log_device.csv $D/runs/${T}_dev.csv
  python3 $D/table.py $T
else
  grep "^CP\|PASSEDCP\|CP {" $LOG | sed 's/^PASSED//' | tail -60
fi
