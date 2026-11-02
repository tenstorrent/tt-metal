#!/bin/bash
# Usage: FILL_VARIANTS=baseline,direct_x0 FILL_SHAPE=8x512 FILL_X=L1 run.sh
# One op per variant, in order; the session's profile log is copied and split by run host ID.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
LOG=/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv
cd "$REPO" && source python_env/bin/activate
mkdir -p "$HERE/_logs"
tag="${FILL_SHAPE:-8x512}_${FILL_X:-L1}_$(date +%H%M%S)"
scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_fill.py" --import-mode=prepend 2>&1 \
    | grep -E "FILL_RESULT|passed|failed|FAILED|Error|SAFE_PYTEST_RESULT" | grep -v "pcc=nan"
cp "$LOG" "$HERE/_logs/$tag.csv"
python3 "$HERE/summarize.py" "$HERE/_logs/$tag.csv" "${FILL_VARIANTS:-baseline}" "$tag" | tee -a "$HERE/results.log"
