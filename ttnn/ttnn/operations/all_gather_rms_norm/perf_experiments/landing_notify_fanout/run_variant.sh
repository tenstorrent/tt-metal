#!/bin/bash
# usage: run_variant.sh <tag> <out_dir> [ENV=VAL ...]   (variant knobs + sweep overrides as env assignments)
# Runs perf_notify.py on the agrms_copy under the eval CCL harness (5 experiments unless CCL_PERF_EXPERIMENTS set).
cd "$(dirname "$0")/../../../../../.."
source python_env/bin/activate
D=ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/landing_notify_fanout
tag=$1; out=$2; shift 2
case "$out" in /*) ;; *) out=$D/$out ;; esac  # relative out dirs live in this experiment dir
mkdir -p "$out"
env "$@" NOTIFY_TAG=$tag CCL_PERF_EXPERIMENTS=${CCL_PERF_EXPERIMENTS:-5} CCL_PERF_OUT=$out \
    scripts/run_safe_pytest.sh --no-precompile $D/perf_notify.py -s --rootdir $PWD/$D 2>&1 |
    grep -E "GEOM chip .0, 0.|fastest chip|slowest chip|spread|per chip|dispatch|PASSED|FAILED|Error|error:|SAFE_PYTEST_RESULT|SKIPPED|assert" | grep -v "^\s*$" | head -30
