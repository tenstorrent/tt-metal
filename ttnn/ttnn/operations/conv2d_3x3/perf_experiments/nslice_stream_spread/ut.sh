#!/usr/bin/env bash
# One run_safe_pytest session of a test file (not profiled), giving up (exit 3) if the shared device lock is not
# acquired within NSS_E_LOCK_WAIT seconds.   ut.sh <tag> <test file> [extra run_safe_pytest flags / pytest args]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; FILE="$2"; shift 2
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
mkdir -p "$HERE/runs"
LOG="$HERE/runs/${TAG}.log"
scripts/run_safe_pytest.sh --run-all "$@" "$FILE" --import-mode=prepend -q > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${NSS_E_LOCK_WAIT:-240} )); then
            pkill -P $PID 2>/dev/null
            kill $PID 2>/dev/null
            echo "LOCK_TIMEOUT after $(( $(date +%s) - START ))s (not started)"
            exit 3
        fi
    fi
    sleep 3
done
wait $PID
grep -E "passed|failed|SAFE_PYTEST_RESULT|^FAILED|HANG|triage|Error" "$LOG" | grep -v "^PRECOMPILE\|waiting for device" | tail -15
