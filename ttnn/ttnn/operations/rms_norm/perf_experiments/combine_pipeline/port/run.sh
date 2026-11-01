#!/bin/bash
# run.sh <tag> <cells> <variants> [profile=1]: one bench session (foreground). Copies the profiler CSVs into runs/
# in the same command (generated/profiler is shared with sibling agents).
cd /localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
source python_env/bin/activate
D=ttnn/ttnn/operations/rms_norm/perf_experiments/combine_pipeline/port
T=$1; CELLS=$2; VARS=$3; PROF=${4:-1}
rm -f $D/runs/${T}_labels.jsonl
LOG=$D/runs/$T.log
PF=""; [ "$PROF" = 1 ] && PF="--profile"
out=$(CP_CELLS=$CELLS CP_VARIANTS=$VARS CP_TAG=$T scripts/run_safe_pytest.sh $PF $EXTRA $D/bench_port.py -s --import-mode=prepend 2>&1)
rc=$?
echo "$out" > $LOG
if [ "$PROF" = 1 ]; then
  csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p')
  [ -n "$csv" ] && cp "$csv" $D/runs/${T}_ops.csv && cp generated/profiler/.logs/profile_log_device.csv $D/runs/${T}_dev.csv
fi
echo "rc=$rc"
echo "$out" | grep -E "SAFE_PYTEST_RESULT|passed|failed|triage report|Error|error:" | tail -8
if [ "$PROF" = 1 ] && [ -f $D/runs/${T}_ops.csv ]; then python3 $D/table.py $T; else grep -o "CP {.*" $LOG | tail -80; fi
