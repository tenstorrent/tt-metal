#!/bin/bash
# Usage: MT_VARIANTS=... MT_MODES=... MT_SHAPES=... run.sh
# Runs the bench under the profiler and joins order.log with the newest ops CSV of this run.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
REPORTS=/localdev/kgrujcic/tt-metal/generated/profiler/reports
rm -f "$HERE/order.log"
marker=$(mktemp)
cd "$REPO" && source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_bench.py" --import-mode=append -s 2>&1 | grep -E "passed|failed|PASSED|FAILED|Error|error:|assert|SAFE_PYTEST|pcc=" | tail -15
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$HERE/last_profile_log.csv"
python3 "$HERE/parse.py" "$marker" "$HERE/order.log" | tee -a "$HERE/results.log"
rm -f "$marker"
