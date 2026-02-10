#!/bin/bash
# dev-test.sh - Run pytest with full debug instrumentation and automatic hang recovery
#
# Enables watcher, lightweight asserts, LLK asserts, and automatic triage on hang.
# Idempotent: handles stale processes, device resets, and leaves the system ready
# for the next invocation regardless of outcome.
#
# Usage: .claude/scripts/dev-test.sh <test_path> [extra_pytest_args...]
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failure (normal pytest failure, no hang)
#   2 - Hang detected (triage output included in stderr)
#   3 - Setup error (missing args, etc.)

set -o pipefail

REPO_DIR="/localdev/mstaletovic/tt-metal"
TIMEOUT_SECONDS=5
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "DEV_TEST_ERROR: No test path provided" >&2
    echo "Usage: .claude/scripts/dev-test.sh <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
shift
EXTRA_ARGS="$@"

# --- Pre-flight: clean up any stale state ---
# Kill stale pytest processes (from previous hung runs)
STALE_PIDS=$(pgrep -f "pytest" 2>/dev/null || true)
if [[ -n "$STALE_PIDS" ]]; then
    echo "DEV_TEST: Cleaning up stale pytest processes..." >&2
    pkill -9 -f pytest 2>/dev/null || true
    sleep 1
    # Reset device since stale pytest means previous hang wasn't cleaned up
    echo "DEV_TEST: Resetting device after stale process cleanup..." >&2
    tt-smi -r 2>/dev/null
    sleep 2
fi

cd "$REPO_DIR"
source python_env/bin/activate 2>/dev/null || true

# --- Build environment ---
export TT_METAL_LLK_ASSERTS=1
export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
export TT_METAL_WATCHER=5
export TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_WATCHER_DISABLE_DISPATCH=1

# Hang detection: dispatch layer invokes triage automatically on timeout
export TT_METAL_OPERATION_TIMEOUT_SECONDS="$TIMEOUT_SECONDS"
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress 1>&2"

echo "DEV_TEST: Running with watcher=ON, timeout=${TIMEOUT_SECONDS}s" >&2
echo "DEV_TEST: Test: ${TEST_PATH} ${EXTRA_ARGS}" >&2
echo "========================================" >&2

# --- Run pytest ---
pytest "${TEST_PATH}" ${EXTRA_ARGS} -v 2>&1
exit_code=$?

echo "========================================" >&2

if [[ $exit_code -eq 0 ]]; then
    # Success - no cleanup needed
    echo "DEV_TEST_RESULT: PASS" >&2
    exit 0
fi

# --- Failure or hang ---
# Kill any remaining pytest processes
pkill -9 -f pytest 2>/dev/null || true

if [[ $exit_code -eq 1 ]]; then
    # Exit code 1 = pytest test failure (could be normal failure or watcher error).
    echo "DEV_TEST_RESULT: FAIL (exit code: $exit_code)" >&2
    echo "DEV_TEST: Resetting device..." >&2
    tt-smi -r 2>/dev/null
    sleep 2
    echo "DEV_TEST: Device reset complete. Ready for next run." >&2
    exit 1
fi

# Exit codes > 1 = crashes, signals, or hangs
echo "DEV_TEST_RESULT: HANG/CRASH (exit code: $exit_code)" >&2
echo "DEV_TEST: Triage output (if any) was printed above by dispatch timeout handler." >&2

# Dump watcher log if it exists
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
if [[ -f "$WATCHER_LOG" ]]; then
    echo "" >&2
    echo "=== WATCHER LOG (last 50 lines) ===" >&2
    tail -50 "$WATCHER_LOG" >&2
    echo "=== END WATCHER LOG ===" >&2
fi

# Full cleanup
echo "DEV_TEST: Resetting device..." >&2
tt-smi -r 2>/dev/null
sleep 2
echo "DEV_TEST: Device reset complete. Ready for next run." >&2
exit 2
