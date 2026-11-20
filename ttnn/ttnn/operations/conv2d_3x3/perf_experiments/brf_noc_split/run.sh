#!/bin/bash
# usage: run.sh <label> <cases> <variants> [extra env...]   e.g. run.sh r1 35,33,41 base,split
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]).
# LOCKWAIT (s, default 400): give up (exit 9, nothing ran) if the shared device lock is not acquired by then; once the
# lock is held the run always completes.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/brf_noc_split
label=$1; cases=$2; variants=$3; shift 3
log=$E/runs/$label.log
: > $log
env BRF_LOG=$label BRF_CASES=$cases BRF_VARIANTS=$variants "$@" scripts/run_safe_pytest.sh --profile --run-all $E/test_brf_split.py > $log 2>&1 &
pid=$!
waited=0
while ! grep -q "Device lock acquired" $log; do
    if ! kill -0 $pid 2>/dev/null; then break; fi
    if [ $waited -ge ${LOCKWAIT:-400} ]; then
        pkill -TERM -P $pid; kill -TERM $pid; wait $pid 2>/dev/null
        echo "RUN_SH: lock not acquired in ${waited}s, gave up"; exit 9
    fi
    sleep 5; waited=$((waited + 5))
done
echo "RUN_SH: lock acquired after ${waited}s"
wait $pid
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|BRF:|BRF_DEBUG|error" | grep -v ccache | sed 's/"t": [0-9.]*, //; s/"knobs": {[^}]*}, //' | cut -c1-240 | tail -40
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" $E/runs/${label}_ops.csv
[ -f generated/profiler/.logs/profile_log_device.csv ] && gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
