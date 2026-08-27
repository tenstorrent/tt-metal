#!/bin/bash
# Reap any process still holding the mesh, and report.
#
# Two scoping rules, both learned the hard way in this session:
#
#  * only signal a PID whose comm is python/python3/pytest. This job runs inside
#    `timeout ... claude -p`, and its own wrapper shells match `pgrep -f pytest`;
#    a previous session in this project killed itself that way.
#  * only signal a PID that actually has /dev/tenstorrent open. An earlier
#    revision signalled every pytest it could see, and so killed a concurrent
#    host-only gate (exit 137, truncated log) that was not touching the mesh at
#    all. Holding a device is the property that matters, not the command line.
set -u

holds_device() {
    local pid="$1"
    ls -l "/proc/$pid/fd" 2>/dev/null | grep -q '/dev/tenstorrent'
}

reap() {
    local signal="$1" found=1
    for p in $(pgrep -f pytest 2>/dev/null); do
        local c; c=$(ps -o comm= -p "$p" 2>/dev/null)
        case "$c" in
            python|python3|pytest) ;;
            *) continue ;;
        esac
        if holds_device "$p"; then
            echo "  reap -$signal $p (comm=$c, holds /dev/tenstorrent)"
            kill "-$signal" "$p" 2>/dev/null
            found=0
        else
            echo "  leave $p (comm=$c, no device open)"
        fi
    done
    return $found
}

if reap TERM; then
    sleep 20
    reap KILL >/dev/null && sleep 10
    echo "REAPED: a device holder survived; the mesh needs a glx_reset before the next run."
    exit 10
fi
echo "mesh free: no device-holding pytest, $(ls /dev/tenstorrent | wc -l) devices"
exit 0
