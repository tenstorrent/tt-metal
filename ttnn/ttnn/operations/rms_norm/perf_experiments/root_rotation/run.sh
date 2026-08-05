#!/usr/bin/env bash
# Run this bench's profiled launches into a PRIVATE profiler tree.
#
# `generated/profiler` is shared with the sibling benches in this clone, so "the newest
# ops_perf_results CSV" is not reliably this run's -- one such collision was caught here
# (a foreign 28-row report vs this bench's 18 launches).  TT_METAL_PROFILER_DIR
# (tools/tracy/common.py) redirects the whole report tree; read_results.py looks there
# first and still verifies the join on row count + input H x W.
#
#   perf_experiments/root_rotation/run.sh -k focus
#   RMS_ROT_VARIANTS=0,1 perf_experiments/root_rotation/run.sh -k "gs32 or gs28"
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export TT_METAL_PROFILER_DIR="${RMS_ROT_PROFILER_DIR:-/tmp/rms_root_rotation_profiler}"
mkdir -p "$TT_METAL_PROFILER_DIR"
exec scripts/run_safe_pytest.sh --profile --run-all \
    ttnn/ttnn/operations/rms_norm/perf_experiments/root_rotation/test_root_rotation.py "$@"
