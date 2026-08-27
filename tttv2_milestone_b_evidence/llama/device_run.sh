#!/bin/bash
# One pytest process on the Galaxy, never piped, one node id per invocation.
# usage: device_run.sh <logfile> <nodeid-or-file> [extra pytest args...]
#
# Why this is not just `timeout ... pytest`:
#
#  * A TT_FATAL/TT_THROW inside a multi-sub-device program leaves the mesh
#    un-drainable. Teardown blocks forever in FDMeshCommandQueue's destructor and
#    the process keeps the per-chip UMD locks, so the *next* run - or any host
#    suite that opens a device - blocks on 'CHIP_IN_USE_<n>_PCIe'. The run must
#    therefore reap itself rather than leaving a holder behind.
#  * It reaps ONLY the pytest process it started, by PID. An earlier revision
#    matched `pgrep -f pytest`, and a still-running reaper from the previous run
#    then SIGKILLed the *next* run's pytest a second after it started (exit 137,
#    empty log). Scope matters more than convenience here.
#  * It never signals a PID whose comm is not python/python3/pytest. This job runs
#    inside `timeout ... claude -p`, and this job's own wrappers match
#    `pgrep -f pytest`; a previous session in this project killed itself that way.
set -u
LOG="$1"; shift
NODE="$1"; shift
export HF_HOME=/proj_sw/user_dev/hf_data
DEADLINE=${MB_DEADLINE:-420}
# attempt 3: the pytest-level timeout is now settable too. A prefill 2048 or an
# 80-layer load legitimately needs longer than 900 s, and a pytest timeout that
# fires early costs a whole run.

{
  echo "=== $(date -u) ==="
  echo "node: $NODE"
  echo "commit: $(git rev-parse HEAD)"
  echo "devices: $(ls /dev/tenstorrent | wc -l)"
} > "$LOG"

python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=${MB_PYTEST_TIMEOUT:-900} "$NODE" "$@" >> "$LOG" 2>&1 &
child=$!

signal_child() {
    local sig="$1" c
    c=$(ps -o comm= -p "$child" 2>/dev/null || true)
    case "$c" in
        python|python3|pytest) echo "  signal -$sig $child (comm=$c)" >> "$LOG"; kill "-$sig" "$child" 2>/dev/null ;;
        *) echo "  refused to signal $child (comm=${c:-gone})" >> "$LOG" ;;
    esac
}

# Two clocks, not one. `DEADLINE` bounds the whole run; `GRACE` bounds how long a
# process may linger *after* pytest has printed its session summary.
#
# A decode-mode failure - a TT_FATAL or a plain AssertionError - leaves the mesh
# un-drainable, and the hang is in the `mesh_device` fixture teardown, after the
# test body and after the summary line. So the verdict is already in the log while
# the process holds all 64 /dev/tenstorrent fds. Waiting out the full deadline for
# a result that has already been written costs ten minutes a run, and attempt 3
# lost a whole run to launching the next cycle while such a holder was still up.
GRACE=${MB_TEARDOWN_GRACE:-90}
waited=0
summary_at=""
while kill -0 "$child" 2>/dev/null && [ "$waited" -lt "$DEADLINE" ]; do
    sleep 5; waited=$((waited + 5))
    # The per-test verdict, not the session summary. Measured on attempt 3's run
    # 23: a decode-mode failure hangs the `mesh_device` fixture teardown *before*
    # pytest gets to write its summary, so the log ends at a bare `FAILED` and a
    # summary-based trigger never fires.
    if [ -z "$summary_at" ] && grep -qE '(^|[[:space:]])(PASSED|FAILED|ERROR)([[:space:]]|$)' "$LOG" 2>/dev/null; then
        summary_at="$waited"
        echo "NOTE: pytest reported a verdict at ${waited}s; teardown grace ${GRACE}s" >> "$LOG"
    fi
    if [ -n "$summary_at" ] && [ $((waited - summary_at)) -ge "$GRACE" ]; then
        echo "TEARDOWN HANG: summary written but the process still holds the mesh after ${GRACE}s" >> "$LOG"
        break
    fi
done

if kill -0 "$child" 2>/dev/null; then
    echo "TIMEOUT after ${DEADLINE}s" >> "$LOG"
    signal_child TERM; sleep 30
    kill -0 "$child" 2>/dev/null && { signal_child KILL; sleep 10; }
    echo "NOTE: reaped on timeout; the mesh needs a glx_reset before the next run." >> "$LOG"
    rc=124
else
    wait "$child"; rc=$?
    # A clean exit code does not mean the process left: pytest can report a
    # failure and then hang in mesh teardown. Only 'gone' means gone.
    if kill -0 "$child" 2>/dev/null; then
        signal_child TERM; sleep 20; signal_child KILL; sleep 5
        echo "NOTE: reaped after exit; the mesh needs a glx_reset." >> "$LOG"
    fi
fi
echo "exit=$rc" >> "$LOG"
echo "holders_after: $(for p in $(pgrep -f pytest 2>/dev/null); do ps -o comm= -p "$p" 2>/dev/null; done | grep -c python)" >> "$LOG"
exit $rc
