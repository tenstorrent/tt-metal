#!/usr/bin/env bash
# Unprofiled pytest session of a test file in this dir (or a ::selection), with the shared device lock wait bounded.
#   [UT_DEV=1] [UT_LOCK_WAIT=s] run_ut.sh <tag> <test path[::sel]> [pytest args...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; shift
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${UT_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
scripts/run_safe_pytest.sh $FLAGS "$@" --import-mode=prepend -q > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${UT_LOCK_WAIT:-300} )); then
            pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null
            echo "LOCK_TIMEOUT after $(( $(date +%s) - START ))s (not started)"; exit 3
        fi
    fi
    sleep 3
done
wait $PID
grep -E "passed|failed|SAFE_PYTEST_RESULT|^FAILED|HANG|triage|Error" "$LOG" | grep -v "^PRECOMPILE" | tail -15
