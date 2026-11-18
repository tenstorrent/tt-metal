#!/bin/bash
# Foreground device run with a lock-wait watchdog: runs scripts/run_safe_pytest.sh with the given arguments, and gives
# up (harmlessly) only while it is still QUEUEING for the shared device lock past $HWR_LOCK_GIVEUP seconds (default
# 480). Output goes to $HWR_LOG (default /tmp/hwr_run.log).
cd /localdev/pmilojevic/workspaces/opgen || exit 9
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
log=${HWR_LOG:-/tmp/hwr_run.log}
giveup=${HWR_LOCK_GIVEUP:-480}
: > "$log"
scripts/run_safe_pytest.sh "$@" > "$log" 2>&1 &
pid=$!
t=0
while kill -0 "$pid" 2>/dev/null; do
    if [ "$t" -ge "$giveup" ] && ! grep -aq "Device lock acquired" "$log"; then
        kill "$pid"
        wait "$pid" 2>/dev/null
        echo "HWR_RUN: gave up waiting for the device lock after ${t}s (never acquired)"
        exit 7
    fi
    sleep 5
    t=$((t + 5))
done
wait "$pid"
rc=$?
grep -a "SAFE_PYTEST_RESULT\|PROFILER CSV\|passed\|failed\| error\|HANG\|triage" "$log" | grep -v "Werror\|Wno-error" | cut -c1-250 | tail -8
exit $rc
