#!/usr/bin/env bash
# usage: wait_idle.sh [max_seconds]  -- poll until no run_safe_pytest.sh / tt-probe.sh session is queued or running
# (the shared device is idle), so the next run starts with its whole timeout budget. Exit 0 idle, 1 still busy.
MAX=${1:-560}; T0=$(date +%s)
while pgrep -f "^(/usr)?/bin/bash ([^ ]*/)?scripts/(run_safe_pytest|tt-probe)\.sh" > /dev/null; do
  (( $(date +%s) - T0 > MAX )) && { echo "busy after ${MAX}s"; exit 1; }
  sleep 3
done
echo "idle after $(( $(date +%s) - T0 ))s"
