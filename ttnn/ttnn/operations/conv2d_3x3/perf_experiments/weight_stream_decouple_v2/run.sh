#!/bin/bash
# usage: run.sh <label> <cases> <variants> [extra env...]   e.g. run.sh first 43,47 base,sp,auto
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]).
# The device lock is shared with sibling experiments: if it is not acquired within WSD_LOCK_WAIT seconds (default 400)
# the still-waiting run is stopped (it never touched the device) and the script exits 75 (re-run it). A run still going
# WSD_DEADLINE seconds after the start (default 570) is stopped with SIGTERM (run_safe_pytest marks the device dirty and
# the next run resets it) and the script exits 124. WSD_NOPROFILE=1 runs without --profile (no timing). WSD_TEST: the
# test file (default test_wsd2.py).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_stream_decouple_v2
label=$1; cases=$2; variants=$3; shift 3
log=$E/runs/$label.log
: > "$log"
setsid env WSD_LOG=$label WSD_CASES=$cases WSD_VARIANTS=$variants "$@" timeout 3000 \
    scripts/run_safe_pytest.sh $([ "${WSD_NOPROFILE:-0}" = 1 ] || echo --profile) --run-all ${WSD_TEST:-$E/test_wsd2.py} > "$log" 2>&1 &
pid=$!
waited=0
while kill -0 $pid 2>/dev/null && ! grep -q "Device lock acquired" "$log"; do
    sleep 5
    waited=$((waited + 5))
    if [ $waited -ge "${WSD_LOCK_WAIT:-400}" ] && ! grep -q "Device lock acquired" "$log"; then
        kill -TERM -- -$pid 2>/dev/null
        wait $pid 2>/dev/null
        echo "RUN.SH: device lock not acquired in ${waited}s; stopped while waiting (re-run)"
        exit 75
    fi
done
while kill -0 $pid 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    if [ $waited -ge "${WSD_DEADLINE:-570}" ]; then
        kill -TERM -- -$pid 2>/dev/null
        wait $pid 2>/dev/null
        echo "RUN.SH: run still going after ${waited}s; stopped (device marked dirty, reset by the next run)"
        tail -5 "$log" | cut -c1-300
        exit 124
    fi
done
wait $pid
out=$(cat "$log")
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|WSD:" | cut -c1-400 | tail -30
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" $E/runs/${label}_ops.csv
[ -n "$csv" ] && [ -f "$(dirname "$csv")/profile_log_device.csv" ] && gzip -c "$(dirname "$csv")/profile_log_device.csv" > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
