#!/bin/bash
# Strictly sequential device cycles, one pytest at a time, from a pipe-delimited
# manifest that also carries a -k selection:
#
#   <wrapper-deadline>|<pytest-deadline>|<logname>|<target>|<extra pytest args>
#
# Nothing starts until the previous cov_run3.sh (pytest + reap + conditional
# glx_reset) has exited.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
while IFS='|' read -r deadline pytimeout logname target extra; do
    case "${deadline:-}" in ''|\#*) continue ;; esac
    MB_DEADLINE="$deadline" MB_PYTEST_TIMEOUT="$pytimeout" MB_EXTRA="${extra:-}" \
        bash "$D/cov_run3.sh" "$logname" "$target" -o faulthandler_timeout=${MB_FAULTHANDLER:-900}
    echo "--- $logname rc=$?  $(date -u +%H:%M:%S)"
done < "$1"
echo "=== sequence complete $(date -u +%H:%M:%S)"
