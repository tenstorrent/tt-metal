#!/bin/bash
# run.sh <tag> <cells> <variants> [profile=1] : one bench session (foreground). Copies the profiler CSVs into runs/
# in the same command (generated/profiler is shared with sibling agents).
cd /localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
source python_env/bin/activate
D=ttnn/ttnn/operations/rms_norm/perf_experiments/row_partial_mcast
T=$1; CELLS=$2; VARS=$3; PROF=${4:-1}
rm -f $D/runs/${T}_labels.txt
LOG=$D/runs/$T.log
PF=""; [ "$PROF" = 1 ] && PF="--profile"
RP_LABELS=runs/${T}_labels.txt RP_CELLS=$CELLS RP_VARIANTS=$VARS scripts/run_safe_pytest.sh $PF --run-all $D/bench_row_partial.py -s --import-mode=prepend > $LOG 2>&1
rc=$?
if [ "$PROF" = 1 ]; then
  csv=$(sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p' $LOG)
  [ -n "$csv" ] && cp "$csv" $D/runs/${T}_ops.csv && cp generated/profiler/.logs/profile_log_device.csv $D/runs/${T}_dev.csv
fi
echo "rc=$rc"
grep -E "SAFE_PYTEST_RESULT|passed|failed|triage report|Error|error:" $LOG | grep -v "^E   \|warning" | tail -6
grep -E "^RP " $LOG | grep -v "pcc_vs_torch=nan"
[ "$PROF" = 1 ] && python3 $D/table.py $T
