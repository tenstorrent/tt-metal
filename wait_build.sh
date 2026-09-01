#!/bin/bash
# Wait for an in-flight build_metal.sh/ninja to finish, then report error count and .so freshness.
cd "$(dirname "$0")"
while pgrep -x ninja > /dev/null 2>&1; do  # -x: exact name, never self-matching command strings
    sleep 15
done
LOG="$1"
echo "errors: $(grep -cE 'error:' "$LOG")"
echo "so: $(date -r build_Release/lib/_ttnn.so '+%H:%M:%S')"
