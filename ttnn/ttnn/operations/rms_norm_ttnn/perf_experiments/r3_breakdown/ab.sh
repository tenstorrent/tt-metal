#!/usr/bin/env bash
# Perf 3 -- drift-cancelling A/B of the whole op: alternate BEFORE and AFTER so the
# 1.5-2.2% first-in-session bias (and any slow session) hits both columns equally.
# BEFORE == a baseline git ref's op files; AFTER == the WORKING TREE as it stands now.
#
# The AFTER state is snapshotted to a temp dir up front and restored from that COPY --
# never with `git checkout HEAD --`, which also reverts UNCOMMITTED work in the same
# files.  Perf 3 lost work to that twice (once in knob.sh, once here: it silently
# reverted D42's ROW_MAJOR carve-out mid-A/B, so both columns measured the same code and
# the "fix" appeared not to work).
set -eu
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
D=$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/r3_breakdown
REL_KERNELS=ttnn/ttnn/operations/rms_norm_ttnn/kernels
REL_DESC=ttnn/ttnn/operations/rms_norm_ttnn/rms_norm_ttnn_program_descriptor.py
BASE=${RMS_BASE_REF:-79ab065a7f}
PROBE=${1:-$D/guard_set.py}
ROUNDS=${RMS_AB_ROUNDS:-2}
cd "$R"
SNAP=$(mktemp -d)
cp -r "$R/$REL_KERNELS" "$SNAP/kernels"
cp "$R/$REL_DESC" "$SNAP/desc.py"
after() { rm -rf "$R/$REL_KERNELS"; cp -r "$SNAP/kernels" "$R/$REL_KERNELS"; cp "$SNAP/desc.py" "$R/$REL_DESC"; }
trap 'after; rm -rf "$SNAP"' EXIT
for r in $(seq 1 "$ROUNDS"); do
  git checkout "$BASE" -- $REL_KERNELS $REL_DESC
  RMS_TAG="BEFORE.r$r" timeout 560 scripts/tt-probe.sh rms_norm_ttnn < "$PROBE" 2>&1 | grep -E "^RESULT"
  after
  RMS_TAG="AFTER.r$r" timeout 560 scripts/tt-probe.sh rms_norm_ttnn < "$PROBE" 2>&1 | grep -E "^RESULT"
done
