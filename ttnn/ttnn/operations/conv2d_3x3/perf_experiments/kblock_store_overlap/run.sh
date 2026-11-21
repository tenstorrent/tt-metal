#!/bin/bash
# usage: run.sh <label> <cases> <variants> [env...]   e.g. run.sh p4a 36 base,whole,free KSO_ITERS=3 KSO_ORDER=alt
#        run.sh <label>                               (keep waiting for / report a run already started)
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / blocking -> runs/<label>.jsonl; KEEP_DEV=1 (default) keeps the device profile
# (runs/<label>_dev.csv.gz) for span_grid.py / zone_report.py; the run's report dir is deleted afterwards. The device lock is shared with sibling experiments, so the
# run is started detached under scripts/run_safe_pytest.sh (which serializes on the lock and watches for hangs) and this
# script waits up to WAIT_S seconds (default 540); exit 75 = still queued or running: call `run.sh <label>` again.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kblock_store_overlap
label=$1
mkdir -p $E/runs
log=$E/runs/$label.log
done=$E/runs/$label.done
if [ $# -ge 3 ]; then
    cases=$2; variants=$3; shift 3
    python3 -I ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r19/zone_hash_check.py $E/op_base $E/op_copy $E/op_final || exit 1
    rm -f $done
    : > $log
    setsid nohup bash -c "env KSO_LOG=$label KSO_CASES=$cases KSO_VARIANTS=$variants $* timeout 3000 \
        scripts/run_safe_pytest.sh --profile --run-all $E/test_kso.py > $log 2>&1; touch $done" > /dev/null 2>&1 &
fi
waited=0
while [ ! -f $done ]; do
    if [ $waited -ge ${WAIT_S:-540} ]; then
        echo "PENDING: $(grep -c 'waiting for device' $log) lock waits, lock acquired: $(grep -c 'Device lock acquired' $log)"
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|KSO:" | sed 's/"knobs": {[^}]*}, //; s/"log": "[^"]*", "t": [0-9.]*, //' | cut -c1-700 | tail -40
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && [ "${KEEP_DEV:-1}" = "1" ] && gzip -c "$(dirname "$csv")/profile_log_device.csv" > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
# The shared disk is nearly full: drop this run's report dir once its CSVs are kept here.
[ -n "$csv" ] && [ -f $E/runs/$label.txt ] && rm -rf "$(dirname "$csv")"
