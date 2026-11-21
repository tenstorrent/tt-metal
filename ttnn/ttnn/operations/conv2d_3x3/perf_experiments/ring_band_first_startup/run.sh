#!/bin/bash
# usage: run.sh <label> <cases> <variants> [env...]   e.g. run.sh t1 42,39 base,bf
#        run.sh <label>                               (keep waiting for / report a run already started)
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / blocking -> runs/<label>.jsonl; KEEP_DEV=1 also keeps the device profile (runs/<label>_dev.csv.gz).
# The device lock is shared with sibling experiments, so the run is started detached (scripts/run_safe_pytest.sh, which
# serializes on the lock) and this script waits for it up to WAIT_S seconds (default 540); exit 75 = still queued or
# running: call `run.sh <label>` again to keep waiting. The menu variants live in op_variants/ (see test_rbf.py).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ring_band_first_startup
label=$1
mkdir -p $E/runs
log=$E/runs/$label.log
done=$E/runs/$label.done
if [ $# -ge 3 ]; then
    # The profiler throws on a 16-bit zone-location hash collision: check before queueing.
    python3 $E/zone_hash_check.py $E/op_base $E/op_copy $E/op_variants | grep -q " 0 collisions" || { python3 $E/zone_hash_check.py $E/op_base $E/op_copy $E/op_variants; exit 3; }
    cases=$2; variants=$3; shift 3
    rm -f $done
    : > $log
    setsid nohup bash -c "env RBF_LOG=$label RBF_CASES=$cases RBF_VARIANTS=$variants $* timeout 3000 \
        scripts/run_safe_pytest.sh --profile --run-all $E/${RBF_TEST:-test_rbf.py} > $log 2>&1; touch $done" > /dev/null 2>&1 &
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
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|RBF:" | sed 's/"knobs": {[^}]*}, //; s/"log": "[^"]*", "t": [0-9.]*, //' | cut -c1-600 | tail -40
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && [ "${KEEP_DEV:-0}" = "1" ] && gzip -c "$(dirname "$csv")/profile_log_device.csv" > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
