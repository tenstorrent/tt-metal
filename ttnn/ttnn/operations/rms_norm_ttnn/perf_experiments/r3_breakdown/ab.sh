#!/usr/bin/env bash
# Perf 3 -- drift-cancelling A/B of the whole op: alternate BEFORE and AFTER so the
# 1.5-2.2% first-in-session bias (and any slow session) hits both columns equally.
# BEFORE == the Perf-2 tip op files; AFTER == the working tree.
set -eu
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
D=$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/r3_breakdown
OP="ttnn/ttnn/operations/rms_norm_ttnn/kernels ttnn/ttnn/operations/rms_norm_ttnn/rms_norm_ttnn_program_descriptor.py"
BASE=${RMS_BASE_REF:-79ab065a7f}
PROBE=${1:-$D/guard_set.py}
ROUNDS=${RMS_AB_ROUNDS:-2}
cd "$R"
restore() { cd "$R" && git checkout HEAD -- $OP; }
trap restore EXIT
for r in $(seq 1 "$ROUNDS"); do
  git checkout "$BASE" -- $OP
  RMS_TAG="BEFORE.r$r" timeout 560 scripts/tt-probe.sh rms_norm_ttnn < "$PROBE" 2>&1 | grep -E "^RESULT"
  git checkout HEAD -- $OP
  RMS_TAG="AFTER.r$r" timeout 560 scripts/tt-probe.sh rms_norm_ttnn < "$PROBE" 2>&1 | grep -E "^RESULT"
done
