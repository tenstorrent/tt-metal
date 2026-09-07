#!/usr/bin/env bash
# Perf 3, Step 1 — CUMULATIVE-PEEL ablation driver.
#
# Stages overlap (NoC reads run concurrently with TRISC compute), so a solo
# removal under-counts.  This peels payloads off CUMULATIVELY by uncommenting the
# shipped kernels' RMS_ABLATE_* defines, measuring, and restoring the sources with
# `git checkout`.  Sync scaffolding (CB reserve/push/wait/pop, loop trip counts,
# zones) stays in place in every configuration — only the payload is stubbed.
#
#   usage: peel.sh "<label>" "<flag> [<flag> ...]"     (flags: READ_X PER_CHANNEL COMPUTE WRITE)
set -u
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
KD=$R/ttnn/ttnn/operations/rms_norm_ttnn/kernels
LABEL="$1"; shift
restore() { cd "$R" && git checkout -- ttnn/ttnn/operations/rms_norm_ttnn/kernels/; }
trap restore EXIT
for f in "$@"; do
  case $f in
    READ_X|PER_CHANNEL) file=$KD/rms_norm_ttnn_reader.cpp ;;
    COMPUTE)            file=$KD/rms_norm_ttnn_compute.cpp ;;
    WRITE|GATHER_ZERO)  file=$KD/rms_norm_ttnn_writer.cpp ;;
    *) echo "unknown flag $f"; exit 1 ;;
  esac
  sed -i "s|^// #define RMS_ABLATE_$f\$|#define RMS_ABLATE_$f|" "$file"
  grep -q "^#define RMS_ABLATE_$f\$" "$file" || { echo "FAILED to enable $f in $file"; exit 1; }
done
cd "$R"
RMS_TAG="$LABEL" timeout 560 scripts/tt-probe.sh rms_norm_ttnn < "$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/r3_breakdown/measure_focus.py" 2>&1 | grep -E "RESULT|Error|error:" | head -20
