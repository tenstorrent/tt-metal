#!/usr/bin/env bash
# A test file of this dir (default test_block_sharded_grad.py) on grad/ (or NT_UT_PKG). Gives up (exit 3) if the
# shared device lock is not acquired within NT_LOCK_WAIT seconds (default 480); never stopped once it holds the device.
#   [NT_DEV=1] [NT_UT_FILE=...] [NT_LOCK_WAIT=s] run_ut.sh <tag> [pytest args]
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
[[ "${NT_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
scripts/run_safe_pytest.sh $FLAGS "$HERE/${NT_UT_FILE:-test_block_sharded_grad.py}" --import-mode=prepend -q "$@" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${NT_LOCK_WAIT:-480} )); then
            pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null
            echo "LOCK_TIMEOUT (not started)"; exit 3
        fi
    fi
    sleep 3
done
wait $PID
grep -E "FAILED|Error|error:|SAFE_PYTEST|passed|failed|Timeout|HANG|triage" "$LOG" | \
    grep -v "^PRECOMPILE\|waiting for device" | tail -30
