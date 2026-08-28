#!/bin/bash
# A single serial device queue, so the mesh is never idle and never shared.
#
# Reads `queue.txt` (same pipe-delimited format as cov_seq2.sh), takes the first
# line, removes it, runs it, repeats. Lines may be appended at any time. Stops
# when the queue is empty AND `queue.stop` exists, or when the file `queue.halt`
# appears (checked between cycles only - a cycle is never interrupted).
#
# Waits at start, and only at start, for any other runner to finish: another
# cov_seq2.sh, a cov_device_run.sh, or a python pytest holding /dev/tenstorrent.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
Q="$D/queue.txt"
REPO="$(cd "$D/../.." && pwd)"
LOG="$D/logs2/queue.out"
busy() {
    # Any other runner, its wrapper, or a reset in flight. `cov_after_device_run.sh`
    # matters as much as the pytest itself: between a pytest exiting and its
    # `tt-smi -glx_reset` finishing there is no device holder and no wrapper, and a
    # second runner that started in that window would reset the mesh under itself.
    for pat in 'cov_seq2\.sh' 'cov_run3_sequence\.sh' 'cov_run3\.sh' 'cov_device_run\.sh' \
               'cov_after_device_run\.sh' 'tt-smi'; do
        pgrep -f "$pat" >/dev/null 2>&1 && { echo "$pat"; return; }
    done
    for p in $(pgrep -f 'python.*-m pytest' 2>/dev/null); do
        c=$(ps -o comm= -p "$p" 2>/dev/null)
        case "$c" in python|python3|pytest) ;; *) continue ;; esac
        ls -l "/proc/$p/fd" 2>/dev/null | grep -q '/dev/tenstorrent' && { echo "pid$p"; return; }
    done
}
{
echo "=== queue runner up $(date -u +%H:%M:%S), waiting for the mesh"
while [ -n "$(busy)" ]; do sleep 20; done
echo "=== queue runner starts $(date -u +%H:%M:%S)"
while true; do
    [ -e "$D/queue.halt" ] && { echo "=== halt file present, stopping $(date -u +%H:%M:%S)"; break; }
    line=$(grep -vE '^\s*(#|$)' "$Q" 2>/dev/null | head -1)
    if [ -z "$line" ]; then
        [ -e "$D/queue.stop" ] && { echo "=== queue empty and stop requested $(date -u +%H:%M:%S)"; break; }
        sleep 30; continue
    fi
    # remove exactly that line, first occurrence
    grep -vxF "$line" "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    while [ -n "$(busy)" ]; do sleep 15; done
    # Disk guard. The device weight cache under model_cache/ is written per
    # device id and a cold weight set costs ~138 GB; /proj_sw started this night
    # 95% full with ~1.0 TB free. Filling a shared filesystem at 04:00 with
    # nobody watching is not an acceptable failure mode, so: prune only the
    # tensorbin files THIS job created (newer than the marker) when space gets
    # tight, and halt rather than continue if that is not enough.
    avail=$(df --output=avail -B1G /proj_sw | tail -1 | tr -d ' ')
    if [ "${avail:-9999}" -lt 300 ]; then
        echo "!!! /proj_sw has ${avail}G free: pruning this job's own cache writes"
        find "$REPO/model_cache" -name '*.tensorbin' -newer "$D/disk_marker" -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | head -400 | cut -d' ' -f2- | while read -r f; do rm -f "$f"; done
        avail=$(df --output=avail -B1G /proj_sw | tail -1 | tr -d ' ')
        echo "!!! after pruning: ${avail}G free"
    fi
    if [ "${avail:-9999}" -lt 150 ]; then
        echo "!!! /proj_sw has only ${avail}G free after pruning; halting the queue rather than filling it"
        break
    fi
    IFS='|' read -r deadline pytimeout logname target extra <<< "$line"
    echo "--- dequeued $logname $(date -u +%H:%M:%S)"
    MB_DEADLINE="$deadline" MB_PYTEST_TIMEOUT="$pytimeout" MB_EXTRA="${extra:-}" \
        TTTV2_GALAXY_CCL_TRACE=${TTTV2_GALAXY_CCL_TRACE:-0} \
        bash "$D/cov_run3.sh" "$logname" "$target" -o faulthandler_timeout=900
    echo "--- $logname rc=$? $(date -u +%H:%M:%S)"
done
echo "=== queue runner done $(date -u +%H:%M:%S)"
} >> "$LOG" 2>&1
