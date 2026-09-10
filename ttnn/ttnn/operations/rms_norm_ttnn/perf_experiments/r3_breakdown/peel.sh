#!/usr/bin/env bash
# Perf 3, Step 1 -- CUMULATIVE-PEEL ablation driver.
#
# Stages overlap (NoC reads run concurrently with TRISC compute), so a SOLO removal
# under-counts: the still-running partner fills the gap.  This peels payloads off
# CUMULATIVELY.  Sync scaffolding (CB reserve/push/wait/pop, loop trip counts, zones)
# stays in place in every configuration -- only the payload is stubbed.
#
# The switches are HOST DEFINES (`RMS_ABLATE=...`), never source edits: the JIT
# kernel cache key does not hash the source's CONTENT, so an in-place edit is a
# cache HIT on the previous build (measured: a "clean" baseline reproduced twice at
# 56,090 ns / pcc=nan against a true 84,510 ns).
#
#   usage: peel.sh "<label>" "READ_X,WRITE,..."      ("" = unablated baseline)
set -eu
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
LABEL="$1"; FLAGS="${2-}"
cd "$R"
RMS_TAG="$LABEL" RMS_ABLATE="$FLAGS" timeout 560 scripts/tt-probe.sh rms_norm_ttnn \
  < "$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/r3_breakdown/measure_focus.py" 2>&1 \
  | grep -E "RESULT|error:" | head -20
