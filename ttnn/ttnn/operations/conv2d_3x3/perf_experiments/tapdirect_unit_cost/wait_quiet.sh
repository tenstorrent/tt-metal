#!/bin/bash
# Wait (up to $1 s, default 540) until no other run_safe_pytest.sh is queued or running: the tool call that launches a
# device run must finish inside its own time limit, so start one only when the device queue is empty.
lim=${1:-540}; t=0
while true; do
  n=$(ps aux | grep "scripts/run_safe_pytest.sh" | grep -v grep | grep -v " timeout " | wc -l)
  if [ "$n" -eq 0 ]; then echo "quiet after ${t}s"; exit 0; fi
  if [ "$t" -ge "$lim" ]; then echo "still busy ($n) after ${t}s"; exit 1; fi
  sleep 15; t=$((t+15))
done
