#!/bin/bash
# Usage: GCO_VARIANTS=a,b GCO_SHAPES=8x512,2x512 run.sh   (profiled run + CSV join)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
rm -f "$HERE/gen/calls.txt"
cd "$REPO" && source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_gco.py" --import-mode=importlib --rootdir="$HERE" \
    --confcutdir="$REPO" -c "$REPO/pytest.ini" -s 2>&1 | grep -E "GCO_RESULT|passed|failed|Error|error:|SAFE_PYTEST_RESULT|triage"
python3 "$HERE/parse.py" | tee -a "$HERE/results.log"
