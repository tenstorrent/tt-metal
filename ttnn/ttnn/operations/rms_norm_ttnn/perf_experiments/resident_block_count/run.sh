#!/usr/bin/env bash
# usage: run.sh "<tag>" [KEY=VAL ...]        -- env-driven variant, foreground, flock'd
# The shipped op is NEVER edited: the variant is selected purely by env vars read by
# the FORKED descriptor in this dir.
set -u
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
D=$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/resident_block_count
TAG="$1"; shift
cd "$R"
env RMS_TAG="$TAG" "$@" timeout 3000 scripts/tt-probe.sh rms_norm_ttnn < "$D/bench.py" 2>&1 \
  | grep -E "RESULT|RMS_BLOCKING|RMS_RBC|error:|Error|FAILED" | head -80
