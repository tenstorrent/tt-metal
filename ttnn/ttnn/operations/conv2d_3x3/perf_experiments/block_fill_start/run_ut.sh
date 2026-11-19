#!/usr/bin/env bash
# Unit tests on grad/: BFS_UT_FILE (default test_block_sharded_grad.py). [BFS_DEV=1] run_ut.sh <tag> [pytest args]
# Gives up (exit 3) if the shared device lock is not acquired within BFS_LOCK_WAIT seconds (default 420).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; shift
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
mkdir -p "$HERE/runs"
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${BFS_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
scripts/run_safe_pytest.sh $FLAGS "${BFS_UT_PATH:-$HERE/${BFS_UT_FILE:-test_block_sharded_grad.py}}" --import-mode=prepend -q "$@" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${BFS_LOCK_WAIT:-420} )); then
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
