#!/bin/bash
# Wait for the mesh to be free of any *other* device run, then run a manifest.
# "Free" means: no cov_seq2.sh/cov_run3.sh other than this script's own tree, and
# no python pytest process with /dev/tenstorrent open. Polls, never signals.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
MAN="$1"
MAXWAIT=${MAXWAIT:-14400}
waited=0
holder() {
    for p in $(pgrep -f 'python.*-m pytest' 2>/dev/null); do
        c=$(ps -o comm= -p "$p" 2>/dev/null)
        case "$c" in python|python3|pytest) ;; *) continue ;; esac
        ls -l "/proc/$p/fd" 2>/dev/null | grep -q '/dev/tenstorrent' && { echo "$p"; return; }
    done
    # a running cov_run3.sh/cov_device_run.sh belonging to another chain
    pgrep -f 'cov_device_run.sh' >/dev/null 2>&1 && { echo "wrapper"; return; }
}
while [ -n "$(holder)" ] && [ "$waited" -lt "$MAXWAIT" ]; do sleep 20; waited=$((waited+20)); done
echo "=== chain start $(basename "$MAN") after ${waited}s wait $(date -u +%H:%M:%S)"
bash "$D/cov_seq2.sh" "$MAN"
