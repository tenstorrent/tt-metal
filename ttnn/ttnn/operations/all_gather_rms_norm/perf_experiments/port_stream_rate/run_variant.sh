#!/bin/bash
# usage: run_variant.sh <tag> <out_dir> [ENV=VAL ...]   (variant knobs + sweep overrides as env assignments)
# Runs perf_psr.py on the agrms_copy under the eval CCL harness (5 experiments unless CCL_PERF_EXPERIMENTS set).
HERE=$PWD; cd "$(dirname "$0")/../../../../../.."; OLDPWD=$HERE
source python_env/bin/activate
D=ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/port_stream_rate
tag=$1; out=$(cd "$OLDPWD" && mkdir -p "$2" && cd "$2" && pwd); shift 2
mkdir -p "$out"
env "$@" PSR_TAG=$tag CCL_PERF_EXPERIMENTS=${CCL_PERF_EXPERIMENTS:-5} CCL_PERF_OUT=$out \
    scripts/run_safe_pytest.sh --no-precompile $D/perf_psr.py -s --rootdir $PWD/$D 2>&1 |
    grep -E "fastest chip|slowest chip|spread|per chip|dispatch|PASSED|FAILED|Error|error:|SAFE_PYTEST_RESULT|SKIPPED|assert" | grep -v "^\s*$" | head -30
