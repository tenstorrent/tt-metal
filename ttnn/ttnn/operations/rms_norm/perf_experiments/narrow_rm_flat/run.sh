#!/bin/bash
# run.sh <tag> <cells> <variants> [profile=1] [defines]: one bench session; copies the profiler CSVs into runs/ in
# the same command (generated/profiler is shared with sibling agents), then prints the table.
cd /localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
source python_env/bin/activate
D=ttnn/ttnn/operations/rms_norm/perf_experiments/narrow_rm_flat
T=$1; CELLS=$2; VARS=$3; PROF=${4:-1}; DEFS=${5:-}
rm -f $D/runs/${T}_labels.jsonl
LOG=$D/runs/$T.log
PF=""; [ "$PROF" = 1 ] && PF="--profile"
NF_CELLS=$CELLS NF_VARIANTS=$VARS NF_TAG=$T NF_DEFINES=$DEFS scripts/run_safe_pytest.sh --run-all ${NF_DEV:-} $PF $D/bench_narrow_rm_flat.py -s --import-mode=prepend > $LOG 2>&1
rc=$?
if [ "$PROF" = 1 ]; then
  csv=$(sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p' $LOG)
  [ -n "$csv" ] && cp "$csv" $D/runs/${T}_ops.csv && cp generated/profiler/.logs/profile_log_device.csv $D/runs/${T}_dev.csv
fi
echo "rc=$rc"
grep -E "SAFE_PYTEST_RESULT|[0-9]+ passed|[0-9]+ failed|triage report|^E  " $LOG | tail -12
python3 $D/table.py $T $PROF
