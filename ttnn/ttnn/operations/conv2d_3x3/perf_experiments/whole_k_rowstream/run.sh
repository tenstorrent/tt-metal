#!/bin/bash
# usage: run.sh <label> <cases> <variants> [env...]   e.g. run.sh t1 41,36 base,rs RS_ORDER=alt
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / ulp / plan -> runs/<label>.jsonl. ABL=<tokens> sets CONV2D_3X3_ABLATE (with RS_ABL=1 variants).
# The device lock is shared with sibling experiments, so the run keeps its place in the lock queue: run.sh starts it
# and follows it (wait.sh) for up to WAIT_S seconds (default 560); if it is still queued or running then, re-run
# wait.sh <label> to keep following it (exit 75 until it is done).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/whole_k_rowstream
label=$1; cases=$2; variants=$3; shift 3
mkdir -p $E/runs
log=$E/runs/$label.log
rm -f $E/runs/$label.done
: > $log
setsid bash -c "env CONV2D_3X3_ABLATE=${ABL:-} RS_LOG=$label RS_CASES=$cases RS_VARIANTS=$variants $* timeout 3000 \
    scripts/run_safe_pytest.sh ${RS_DEV:+--dev} --profile --run-all $E/test_rs.py > $log 2>&1; \
    ${DEVCSV:+gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/${label}_dev.csv.gz;} touch $E/runs/$label.done" \
    < /dev/null > /dev/null 2>&1 &
exec $E/wait.sh $label
