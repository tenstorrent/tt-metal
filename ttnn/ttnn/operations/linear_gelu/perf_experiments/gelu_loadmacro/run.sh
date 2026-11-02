#!/bin/bash
# Usage: [--profile] run.sh <test_file>[::test] [extra pytest args]   (env GV_VARIANTS / GV_REPS / GV_TILES)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
PROF=""
if [ "$1" == "--profile" ]; then PROF="--profile"; shift; fi
T="$1"; shift
cd "$REPO" && source python_env/bin/activate
scripts/run_safe_pytest.sh $PROF --run-all "$HERE/$T" --import-mode=importlib --rootdir="$HERE" \
    --confcutdir="$REPO" -c "$REPO/pytest.ini" -s "$@" 2>&1 | grep -v "riscv-tt-elf" | grep -E "PRECISION|LAUNCH|^E2E|passed|failed|Error|error|SAFE_PYTEST_RESULT|triage|assert"
