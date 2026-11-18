#!/bin/bash
# Waits (outside any device session) until the shared device lock is free, at most $1 seconds (default 300), so the
# following foreground run_safe_pytest.sh call spends its tool-time budget running rather than queueing.
limit=${1:-300}
t=0
while ! flock -n /tmp/tt-device.lock true 2>/dev/null; do
    sleep 3
    t=$((t + 3))
    if [ "$t" -ge "$limit" ]; then
        echo "lock still busy after ${t}s"
        exit 1
    fi
done
echo "lock free after ${t}s"
