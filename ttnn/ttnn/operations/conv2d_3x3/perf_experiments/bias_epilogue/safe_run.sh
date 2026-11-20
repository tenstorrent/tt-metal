#!/bin/bash
# usage: safe_run.sh <logfile> <run_safe_pytest args...>
# Runs scripts/run_safe_pytest.sh and stays attached until it exits, but gives up (exit 75, "DEFERRED") if the device
# lock is not acquired within BE_MAXWAIT seconds (default 300): a run killed by the caller's time limit while it holds
# the device can wedge it for the next runner; a run killed while still queued on the flock holds nothing.
log=$1; shift
maxwait=${BE_MAXWAIT:-300}
scripts/run_safe_pytest.sh "$@" > "$log" 2>&1 &
pid=$!
t=0
while kill -0 $pid 2>/dev/null; do
  if ! grep -q "SAFE_PYTEST: Device lock acquired" "$log"; then
    if [ "$t" -ge "$maxwait" ]; then
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
      echo "DEFERRED: device lock not acquired in ${maxwait}s"
      exit 75
    fi
  fi
  sleep 5; t=$((t+5))
done
wait $pid
