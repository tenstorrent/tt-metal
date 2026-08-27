#!/bin/bash
# One full device cycle: run, report, clean up. Always launch this in the
# background - a reset plus a run plus a reset exceeds the 600 s foreground cap,
# and a foreground timeout SIGTERMs the process group mid-run.
# usage: cycle.sh <logfile> <nodeid> [pytest args...]
set -u
D="$(dirname "$0")"
LOG="$1"; shift
bash "$D/device_run.sh" "$LOG" "$@"
rc=$?
bash "$D/after_device_run.sh" "$(basename "$LOG" .log)" "$rc"
exit $rc
