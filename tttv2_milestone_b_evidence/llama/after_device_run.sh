#!/bin/bash
# Always run this after a device run: reap a hung teardown and reset the mesh if
# anything had to be killed. A TT_FATAL leaves the mesh un-drainable and the
# process keeps the per-chip UMD locks, which blocks the *next* run and any host
# suite that opens a device (tests/models/galaxy/test_plans.py does).
set -u
TAG="${1:-cleanup}"
RC="${2:-0}"   # exit code of the device run; non-zero => always reset
out=$(bash "$(dirname "$0")/ensure_mesh_free.sh")
echo "$out"
# Reset after *any* non-clean run, not only when a holder had to be reaped. An
# aborted multi-sub-device program leaves the fabric dirty, and the next mesh
# open then intermittently fails topology discovery with
#   Timed out waiting for ETH heartbeat on device ASIC ID ... ETH core e9-0
# which costs a whole run to rediscover.
if echo "$out" | grep -q REAPED || [ "$RC" != "0" ]; then
    R="$(dirname "$0")/logs/reset_${TAG}.log"
    { date -u; echo "reset after reaping a hung teardown ($TAG)"; } > "$R"
    # 900, not 600. A glx_reset after a wedged ARC controller needs longer than
    # ten minutes on this machine: attempt 2's reset after run 07 was killed at
    # the 600 s cap inside `Issuing POST_RESET`, which left a chip's ARC firmware
    # half-initialised (`ARC startup error at core 0-10 over NOC0 ... Timed out
    # after 300000 ms`) and cost a full recovery cycle. The same reset with 900 s
    # succeeded. The cap is most needed in exactly the case it was too tight for.
    timeout 900 tt-smi -glx_reset >> "$R" 2>&1
    echo "reset exit=$?" >> "$R"
    grep -oE 'Re-initialized 32 boards after reset|Error in resetting[^ ]*' "$R" | tail -1
fi
echo "devices=$(ls /dev/tenstorrent | wc -l)"
