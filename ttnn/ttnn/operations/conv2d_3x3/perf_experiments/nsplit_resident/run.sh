#!/bin/bash
# usage: run.sh <label> <cases> <variants> [env...]   e.g. run.sh t1 43,38,47 base,ns
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / blocking -> runs/<label>.jsonl.
# The device lock is shared with sibling experiments: if it is not acquired within WAIT_S seconds (default 330) the
# queued run (still waiting on the flock, never on the device) is stopped and the script exits 75 (re-run it). Once the
# lock is held the run always completes.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nsplit_resident
label=$1; cases=$2; variants=$3; shift 3
mkdir -p $E/runs
log=$E/runs/$label.log
: > $log
setsid env NSR_LOG=$label NSR_CASES=$cases NSR_VARIANTS=$variants "$@" timeout 3000 \
    scripts/run_safe_pytest.sh --profile --run-all $E/test_nsr.py > $log 2>&1 &
pid=$!
waited=0
while ! grep -q "Device lock acquired" $log; do
    if ! kill -0 $pid 2>/dev/null; then break; fi
    if [ $waited -ge ${WAIT_S:-330} ]; then
        kill -TERM -- -$pid 2>/dev/null
        wait $pid 2>/dev/null
        echo "RETRY: device lock not acquired in ${waited}s ($(grep -c 'waiting for device' $log) waits)"
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done
wait $pid
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|NSR:" | sed 's/"knobs": {[^}]*}, //; s/"log": "[^"]*", "t": [0-9.]*, //' | cut -c1-600 | tail -40
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
