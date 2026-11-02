#!/bin/bash
# Usage: [GST_VARIANTS=a,b] run.sh <test_file.py>[::test] [--profile]
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
T="$1"; shift
cd "$REPO" && source python_env/bin/activate
scripts/run_safe_pytest.sh "$@" --run-all "$HERE/$T" --import-mode=importlib --rootdir="$HERE" \
    --confcutdir="$REPO" -c "$REPO/pytest.ini" -s 2>&1 | grep -v "cmd:" | grep -E "GST_|passed|failed| error:|SAFE_PYTEST_RESULT|triage|Timeout|hang" | cut -c1-400 | sort -u | head -60
