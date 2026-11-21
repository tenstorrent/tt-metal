#!/bin/bash
# usage: safe_run.sh <logfile> <run_safe_pytest args...>
# Runs scripts/run_safe_pytest.sh and stays attached until it exits, but gives up (exit 75, "DEFERRED") if the device
# lock is not acquired within BE_MAXWAIT seconds (default 300): a run killed by the caller's time limit while it holds
# the device can wedge it for the next runner; a run killed while still queued on the flock holds nothing.
# TU_COPY_DEV=<path>: copy the device profile log as soon as the run reports its CSV (it still holds the lock then; the
# next profiled run on the machine clears the log).
log=$1; shift
maxwait=${BE_MAXWAIT:-300}
scripts/run_safe_pytest.sh "$@" > "$log" 2>&1 &
pid=$!
t=0
copied=0
while kill -0 $pid 2>/dev/null; do
  if ! grep -q "SAFE_PYTEST: Device lock acquired" "$log"; then
    if [ "$t" -ge "$((maxwait * 5))" ]; then
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
      echo "DEFERRED: device lock not acquired in ${maxwait}s"
      exit 75
    fi
  fi
  if [ -n "$TU_COPY_DEV" ] && [ $copied -eq 0 ] && grep -q "PROFILER CSV:" "$log"; then
    cp generated/profiler/.logs/profile_log_device.csv "$TU_COPY_DEV" 2>/dev/null && copied=1
  fi
  sleep 0.2; t=$((t+1))
done
if [ -n "$TU_COPY_DEV" ] && [ $copied -eq 0 ]; then
  cp generated/profiler/.logs/profile_log_device.csv "$TU_COPY_DEV" 2>/dev/null
fi
wait $pid
