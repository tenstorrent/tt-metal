#!/bin/bash
# Foreground device run with a lock-wait watchdog: runs scripts/run_safe_pytest.sh with the given arguments, and gives
# up (harmlessly) only while it is still QUEUEING for the shared device lock past $HBF_LOCK_GIVEUP seconds (default
# 420), so the caller's tool-time budget never ends mid-run. Output goes to $HBF_LOG (default /tmp/hbf/run.log).
cd /localdev/pmilojevic/workspaces/opgen || exit 9
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
log=${HBF_LOG:-/tmp/hbf/run.log}
giveup=${HBF_LOCK_GIVEUP:-420}
: > "$log"
scripts/run_safe_pytest.sh "$@" > "$log" 2>&1 &
pid=$!
t=0
while kill -0 "$pid" 2>/dev/null; do
    if [ "$t" -ge "$giveup" ] && ! grep -aq "Device lock acquired" "$log"; then
        kill "$pid"
        wait "$pid" 2>/dev/null
        echo "HBF_RUN: gave up waiting for the device lock after ${t}s (never acquired)"
        exit 7
    fi
    sleep 5
    t=$((t + 5))
done
wait "$pid"
rc=$?
grep -a "SAFE_PYTEST_RESULT\|PROFILER CSV\|passed\|failed\| error" "$log" | grep -v "Werror\|Wno-error" | cut -c1-250 | tail -6
exit $rc
