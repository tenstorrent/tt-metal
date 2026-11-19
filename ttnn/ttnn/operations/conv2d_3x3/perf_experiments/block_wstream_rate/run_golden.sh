#!/usr/bin/env bash
# The golden suite on grad2/ (test_golden_grad.py), one -k chunk per call: run_golden.sh <tag> "<-k expr>"
# [BWR_GOLDEN_PKG=op_base] for HEAD. Gives up (exit 3) if the device lock is not acquired within BWR_LOCK_WAIT s.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; KEXPR="$2"
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/golden_${TAG}.log"
scripts/run_safe_pytest.sh --run-all --precompile "$HERE/test_golden_grad.py" --import-mode=prepend -q -k "$KEXPR" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${BWR_LOCK_WAIT:-420} )); then
            pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null
            echo "LOCK_TIMEOUT after $(( $(date +%s) - START ))s (not started)"; exit 3
        fi
    fi
    sleep 3
done
wait $PID
grep -E "^FAILED|passed|failed|SAFE_PYTEST_RESULT|SAFE_PYTEST_TOTAL|HANG|triage" "$LOG" | grep -v "waiting for device" | tail -15
