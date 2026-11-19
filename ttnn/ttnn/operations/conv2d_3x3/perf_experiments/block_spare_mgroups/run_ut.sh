#!/usr/bin/env bash
# The BLOCK unit tests (test_block_sharded_grad.py) on grad/ (or BSM_UT_PKG), both fidelities. [BSM_DEV=1] run_ut.sh <tag> [pytest args]
# Gives up (exit 3) if the shared device lock is not acquired within BSM_LOCK_WAIT seconds (default 420).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; shift
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${BSM_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
scripts/run_safe_pytest.sh $FLAGS "$HERE/${BSM_UT_FILE:-test_block_sharded_grad.py}" --import-mode=prepend -q "$@" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${BSM_LOCK_WAIT:-420} )); then
            pkill -P $PID 2>/dev/null
            kill $PID 2>/dev/null
            echo "LOCK_TIMEOUT after $(( $(date +%s) - START ))s (not started)"
            exit 3
        fi
    fi
    sleep 3
done
wait $PID
grep -E "FAILED|Error|error:|SAFE_PYTEST|passed|failed|Timeout|HANG|triage" "$LOG" | \
    grep -v "^PRECOMPILE\|waiting for device" | tail -30
