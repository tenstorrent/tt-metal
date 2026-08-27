#!/bin/bash
# Run a list of device cycles strictly sequentially, one pytest on the mesh at a
# time, each followed by cycle.sh's reap-and-reset. Never pipes pytest.
#
# usage: run_sequence.sh <manifest>
#   manifest lines:  <deadline-seconds> <logname> <pytest-node-id>
#   blank lines and lines starting with # are skipped.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$1"
export HF_HOME=/proj_sw/user_dev/hf_data

while read -r deadline logname node; do
    case "${deadline:-}" in ''|\#*) continue ;; esac
    LOG="$D/logs2/${logname}.log"
    echo "=== $(date -u +%H:%M:%S) start $logname (deadline ${deadline}s) node=$node"
    MB_DEADLINE="$deadline" bash "$D/cycle.sh" "$LOG" "$node"
    rc=$?
    echo "=== $(date -u +%H:%M:%S) end   $logname rc=$rc"
    # A summary line per run so the driver log alone tells the story.
    grep -oE '[0-9]+ (passed|failed|error)[^,)]*' "$LOG" | tail -3 | tr '\n' ' '
    echo
done < "$MANIFEST"
echo "=== sequence complete"
