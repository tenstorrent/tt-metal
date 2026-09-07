#!/usr/bin/env bash
# Perf 3 — flip ONE module-level knob in the SHIPPED descriptor, measure, restore.
#   usage: knob.sh "<label>" "<PY_NAME> = <value>" [more assignments...]
set -u
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
F=$R/ttnn/ttnn/operations/rms_norm_ttnn/rms_norm_ttnn_program_descriptor.py
LABEL="$1"; shift
restore() { cd "$R" && git checkout -- ttnn/ttnn/operations/rms_norm_ttnn/rms_norm_ttnn_program_descriptor.py; }
trap restore EXIT
for a in "$@"; do
  name="${a%% =*}"
  sed -i "0,/^$name = /{s|^$name = .*|$a|}" "$F"
  grep -qxF "$a" "$F" || { echo "FAILED to set: $a"; exit 1; }
done
cd "$R"
RMS_TAG="$LABEL" timeout 560 scripts/tt-probe.sh rms_norm_ttnn < "$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/r3_breakdown/measure_focus.py" 2>&1 | grep -E "RESULT|error:" | head -20
