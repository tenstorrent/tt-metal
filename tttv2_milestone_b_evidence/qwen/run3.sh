#!/bin/bash
# attempt 3 wrapper: one device cycle into logs2/, with the CCL trace on and a
# settable pytest-level timeout. Never pipes pytest.
# usage: MB_DEADLINE=1500 MB_PYTEST_TIMEOUT=1200 run3.sh <logname> <nodeid> [pytest args...]
set -u
D="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME=/localdev/ctr-apbernal/hf_data
# The CCL trace also *synchronizes* after each LM head collective op, which is
# what named D-B19 - and three extra device syncs per token is a real wall-clock
# cost on a 511-step decode. Respect a value the caller already set so a long run
# can turn it off.
export TTTV2_GALAXY_CCL_TRACE=${TTTV2_GALAXY_CCL_TRACE-1}
export MB_RESET_DIR="$D/logs2"
REPO="$(cd "$D/../.." && pwd)"
cd "$REPO" || exit 1
NAME="$1"; shift
LOG="$D/logs2/${NAME}.log"
echo "=== $(date -u +%H:%M:%S) start $NAME deadline=${MB_DEADLINE:-420}s node=$1"
bash "$D/device_run.sh" "$LOG" "$@"
rc=$?
bash "$D/after_device_run.sh" "$NAME" "$rc"
echo "=== $(date -u +%H:%M:%S) end $NAME rc=$rc"
grep -oE '[0-9]+ (passed|failed|error)[^,)]*' "$LOG" | tail -3 | tr '\n' ' '; echo
grep -E '^\[stage\]|^\[ccl\]' "$LOG" | tail -25
exit $rc
