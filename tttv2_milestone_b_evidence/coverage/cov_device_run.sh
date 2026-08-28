#!/bin/bash
# One pytest process on the Galaxy, never piped. Accepts a node id, a file, or
# several targets (see MB_EXTRA for -k selection).
#
# Derived from tttv2_milestone_b_evidence/qwen/device_run.sh, with ONE change,
# and it matters: the qwen version arms its teardown-grace timer on the FIRST
# per-test verdict in the log. That is correct for a one-node-id run and wrong
# for a multi-test run - after test 1 prints PASSED, the timer fires 90 s later
# while test 2 is still running, and reaps a healthy process. mb-coverage runs
# whole files (8 node ids in the llama full-model file), so the trigger here is
#
#   * the pytest SESSION summary ("short test summary info" / "=== N passed"),
#     which really is the last thing before fixture teardown; or
#   * log silence: no write for MB_IDLE seconds with at least one verdict
#     already in hand. A hung mesh teardown stops writing; a long prefill does
#     not (ttnn logs continuously).
#
# usage: MB_DEADLINE=5400 MB_PYTEST_TIMEOUT=3000 cov_device_run.sh <logfile> <target> [pytest args...]
set -u
LOG="$1"; shift
NODE="$1"; shift
export HF_HOME=/localdev/ctr-apbernal/hf_data
DEADLINE=${MB_DEADLINE:-420}

{
  echo "=== $(date -u) ==="
  echo "node: $NODE"
  echo "extra: ${MB_EXTRA:-}"
  echo "commit: $(git rev-parse HEAD)"
  echo "HF_HOME: $HF_HOME"
  echo "sys devices: $(ls /sys/class/tenstorrent | wc -l)"
} > "$LOG"

# shellcheck disable=SC2086
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=${MB_PYTEST_TIMEOUT:-900} "$NODE" ${MB_EXTRA:-} "$@" >> "$LOG" 2>&1 &
child=$!

signal_child() {
    local sig="$1" c
    c=$(ps -o comm= -p "$child" 2>/dev/null || true)
    case "$c" in
        python|python3|pytest) echo "  signal -$sig $child (comm=$c)" >> "$LOG"; kill "-$sig" "$child" 2>/dev/null ;;
        *) echo "  refused to signal $child (comm=${c:-gone})" >> "$LOG" ;;
    esac
}

GRACE=${MB_TEARDOWN_GRACE:-120}
IDLE=${MB_IDLE:-900}
waited=0
summary_at=""
while kill -0 "$child" 2>/dev/null && [ "$waited" -lt "$DEADLINE" ]; do
    sleep 5; waited=$((waited + 5))
    if [ -z "$summary_at" ] && grep -qE 'short test summary info|^=+ .*(passed|failed|error|no tests ran)' "$LOG" 2>/dev/null; then
        summary_at="$waited"
        echo "NOTE: pytest wrote its session summary at ${waited}s; teardown grace ${GRACE}s" >> "$LOG"
    fi
    if [ -n "$summary_at" ] && [ $((waited - summary_at)) -ge "$GRACE" ]; then
        echo "TEARDOWN HANG: session summary written but the process still holds the mesh after ${GRACE}s" >> "$LOG"
        break
    fi
    if [ $((waited % 60)) -eq 0 ]; then
        now=$(date +%s); mt=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
        if [ $((now - mt)) -ge "$IDLE" ] && grep -qE '(^|[[:space:]])(PASSED|FAILED|ERROR)([[:space:]]|$)' "$LOG" 2>/dev/null; then
            echo "IDLE HANG: no log write for $((now - mt))s with a verdict already in hand" >> "$LOG"
            break
        fi
    fi
done

if kill -0 "$child" 2>/dev/null; then
    echo "TIMEOUT/REAP after ${waited}s (deadline ${DEADLINE}s)" >> "$LOG"
    signal_child TERM; sleep 30
    kill -0 "$child" 2>/dev/null && { signal_child KILL; sleep 10; }
    echo "NOTE: reaped; the mesh needs a glx_reset before the next run." >> "$LOG"
    rc=124
else
    wait "$child"; rc=$?
    if kill -0 "$child" 2>/dev/null; then
        signal_child TERM; sleep 20; signal_child KILL; sleep 5
        echo "NOTE: reaped after exit; the mesh needs a glx_reset." >> "$LOG"
    fi
fi
echo "exit=$rc" >> "$LOG"
echo "holders_after: $(for p in $(pgrep -f pytest 2>/dev/null); do ps -o comm= -p "$p" 2>/dev/null; done | grep -c python)" >> "$LOG"
exit $rc
