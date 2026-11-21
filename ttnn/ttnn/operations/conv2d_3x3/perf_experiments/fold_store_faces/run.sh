#!/bin/bash
# usage: run.sh <label> <cases> <variants> [env...]   e.g. run.sh t1 35,33 base,f2 FS_ORDER=alt
# Runs in the foreground (the device lock is shared with sibling experiments: the run waits for it). Keep a run well
# under the 10-minute tool limit: a killed --profile run loses its profiler CSV.
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / ulp / plan -> runs/<label>.jsonl. DEVCSV=1 keeps the device-side CSV (zones) gzipped.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fold_store_faces
label=$1; cases=$2; variants=$3; shift 3
mkdir -p $E/runs
log=$E/runs/$label.log
# The run waits in this call; while it still waits for the device lock after LOCK_WAIT_S seconds (default 200) it is
# stopped (no device touched yet) and the script exits 75: run it again.
env CONV2D_3X3_ABLATE=${ABL:-} FS_LOG=$label FS_CASES=$cases FS_VARIANTS=$variants "$@" \
    scripts/run_safe_pytest.sh ${FS_DEV:+--dev} ${FS_NOPROF:---profile} --run-all $E/test_fsf.py > $log 2>&1 &
pid=$!
waited=0
while kill -0 $pid 2>/dev/null && ! grep -q "Device lock acquired" $log; do
    if [ $waited -ge ${LOCK_WAIT_S:-200} ]; then
        pkill -TERM -P $pid; kill -TERM $pid; wait $pid 2>/dev/null
        echo "QUEUED: lock not acquired in ${waited}s; stopped before touching the device"
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done
wait $pid
[ -n "$DEVCSV" ] && gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/${label}_dev.csv.gz
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|waiting for device" | tail -6
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && [ -f "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && [ -f "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
# Keep the run log small: drop the per-op lines.
grep -E "FS:|passed|failed|SAFE_PYTEST|HANG|triage|Error|error|PROFILER" $log > $log.tmp; mv $log.tmp $log
