#!/bin/bash
# Run a list of attempt-3 device cycles strictly sequentially: one pytest on the
# mesh at a time, each followed by run3.sh's reap-and-reset. Never pipes pytest,
# and never starts the next cycle until the previous run3.sh has exited - the
# pytest verdict landing in the log is *not* that moment, because a decode-mode
# failure hangs the mesh_device fixture teardown after it.
#
# usage: run3_sequence.sh <manifest>
#   manifest lines: <wrapper-deadline> <pytest-deadline> <logname> <node-id>
set -u
D="$(cd "$(dirname "$0")" && pwd)"
while read -r deadline pytimeout logname node; do
    case "${deadline:-}" in ''|\#*) continue ;; esac
    MB_DEADLINE="$deadline" MB_PYTEST_TIMEOUT="$pytimeout" \
        bash "$D/run3.sh" "$logname" "$node" -o faulthandler_timeout=${MB_FAULTHANDLER:-600}
    echo "--- $logname rc=$?"
done < "$1"
echo "=== sequence complete"
