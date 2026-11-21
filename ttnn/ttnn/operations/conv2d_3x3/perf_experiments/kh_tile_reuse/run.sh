#!/bin/bash
# usage: run.sh <label> <cases> <variants> [env...]   e.g. run.sh t1 36,41 base,khr
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / blocking -> runs/<label>.jsonl; KEEP_DEV=1 also keeps the device zone log (runs/<label>_dev.csv.gz).
# The device lock is shared with sibling experiments. The job keeps its place on the flock; this script waits at most
# WAIT_S seconds (default 480) for it to finish, else prints PENDING and exits 76: re-run the same command (same label)
# to keep waiting on the same job (it is never launched twice).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kh_tile_reuse
label=$1; cases=$2; variants=$3; shift 3
mkdir -p $E/runs
log=$E/runs/$label.log
pidf=$E/runs/$label.pid
if [ -f $pidf ] && kill -0 $(cat $pidf) 2>/dev/null; then
    pid=$(cat $pidf)
else
    : > $log
    rm -f $E/runs/$label.done
    setsid bash -c "env KHR_LOG=$label KHR_CASES=$cases KHR_VARIANTS=$variants $* timeout 3000 \
        scripts/run_safe_pytest.sh --profile --run-all $E/test_khr.py > $log 2>&1; touch $E/runs/$label.done" \
        > /dev/null 2>&1 < /dev/null &
    pid=$!
    echo $pid > $pidf
fi
waited=0
while [ ! -f $E/runs/$label.done ]; do
    if [ $waited -ge ${WAIT_S:-480} ]; then
        echo "PENDING: $(grep -c 'waiting for device' $log) lock waits, acquired: $(grep -c 'Device lock acquired' $log)"
        exit 76
    fi
    sleep 5
    waited=$((waited + 5))
done
rm -f $pidf
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|KHR:" | sed 's/"knobs": {[^}]*}, //; s/"log": "[^"]*", "t": [0-9.]*, //' | cut -c1-700 | tail -40
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && [ "${KEEP_DEV:-0}" = "1" ] && gzip -c "$(dirname "$csv")/profile_log_device.csv" > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
