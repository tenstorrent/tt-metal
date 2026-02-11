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
#   2 - Hang detected (triage summary printed, full triage at /tmp/dev-test-triage.log)
#   3 - Setup error (missing args, etc.)
#
# NOTE: The dispatch timeout mechanism converts hangs into exceptions that pytest
# catches as normal test failures (exit code 1). Hangs are distinguished from
# normal failures by checking whether triage output was generated (the timeout
# handler runs tt-triage before raising the exception).

set -o pipefail

REPO_DIR="/localdev/mstaletovic/tt-metal"
TIMEOUT_SECONDS=5
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
TRIAGE_SUMMARIZER="${REPO_DIR}/.claude/scripts/summarize-triage.py"
TRIAGE_LOG="/tmp/dev-test-triage.log"

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
# Triage output goes to a separate file to keep pytest output clean
export TT_METAL_OPERATION_TIMEOUT_SECONDS="$TIMEOUT_SECONDS"
rm -f "$TRIAGE_LOG"
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress > ${TRIAGE_LOG} 2>&1"

echo "DEV_TEST: Running with watcher=ON, timeout=${TIMEOUT_SECONDS}s" >&2
echo "DEV_TEST: Test: ${TEST_PATH} ${EXTRA_ARGS}" >&2
echo "========================================" >&2

# --- Run pytest ---
# -x: stop on first failure (avoids running doomed tests after a hang bricks the device)
pytest "${TEST_PATH}" -x ${EXTRA_ARGS} -v 2>&1
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

# Determine if this was a hang: the dispatch timeout handler writes triage output
# to TRIAGE_LOG before raising an exception. If the file exists and is non-empty,
# a hang occurred. Pytest exit code is 1 in both cases (normal fail and hang-as-exception).
is_hang=false
if [[ -s "$TRIAGE_LOG" ]]; then
    is_hang=true
fi

if [[ "$is_hang" == true ]]; then
    echo "DEV_TEST_RESULT: HANG (detected via triage — dispatch timeout fired)" >&2
    echo "" >&2

    # Summarize triage output
    python3 "$TRIAGE_SUMMARIZER" "$TRIAGE_LOG" >&2 2>/dev/null || \
        echo "DEV_TEST: Triage output saved to: ${TRIAGE_LOG} (summarizer failed)" >&2
    echo "" >&2

    # Dump watcher log — useful for waypoints and assert failures
    WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
    if [[ -f "$WATCHER_LOG" ]]; then
        echo "=== WATCHER LOG (last 50 lines) ===" >&2
        tail -50 "$WATCHER_LOG" >&2
        echo "=== END WATCHER LOG ===" >&2
        echo "" >&2
    fi

    echo "DEV_TEST: Resetting device..." >&2
    tt-smi -r 2>/dev/null
    sleep 2
    echo "DEV_TEST: Device reset complete. Ready for next run." >&2
    exit 2
fi

# Normal test failure (no hang) — exit code 1 from pytest
echo "DEV_TEST_RESULT: FAIL (exit code: $exit_code)" >&2
echo "DEV_TEST: Resetting device..." >&2
tt-smi -r 2>/dev/null
sleep 2
echo "DEV_TEST: Device reset complete. Ready for next run." >&2
exit 1
