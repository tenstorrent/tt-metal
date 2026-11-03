#!/bin/bash
# Usage: PT_VARIANTS=a,b PT_SHAPES=8x512 PT_INPUTS=model [--profile] run.sh [test_file]   (+ CSV join when profiling)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
PROF=""
if [ "$1" == "--profile" ]; then PROF="--profile"; shift; fi
T="${1:-test_e2e.py}"
rm -f "$HERE/gen/calls.txt"
export TT_METAL_PROFILER_DIR="$HERE/gen/prof"; mkdir -p "$TT_METAL_PROFILER_DIR"
cd "$REPO" && source python_env/bin/activate
scripts/run_safe_pytest.sh $PROF --run-all "$HERE/$T" --import-mode=importlib --rootdir="$HERE" \
    --confcutdir="$REPO" -c "$REPO/pytest.ini" -s 2>&1 | grep -v riscv-tt-elf | grep -E "PT_RESULT|PT_BENCH|passed|failed|Error|error:|SAFE_PYTEST_RESULT|triage|assert"
if [ -n "$PROF" ]; then python3 "$HERE/parse.py" | tee -a "$HERE/results.log"; fi
