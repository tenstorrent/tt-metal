#!/bin/bash
# Usage: SRM_CASES=a,b SRM_VARIANTS=x,y bench.sh [--profile]
# Prints "<case> <variant> <DEVICE KERNEL DURATION ns>" per GenericOp row, in run order.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/split_reader_mixed
rm -f "$here/run_order.txt"
marker=$(mktemp)
scripts/run_safe_pytest.sh --run-all ${SRM_DEV:-} ${1:-} "$here/test_srm.py" --import-mode=prepend >"$here/last_run.log" 2>&1
grep -E "SRM_RESULT|SAFE_PYTEST_RESULT|FAILED|Error|hang|HANG" "$here/last_run.log" | grep -v "^E " | head -40
if [ "${1:-}" = "--profile" ]; then
    csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv"; exit 1; fi
    vals=$(awk -F',' 'NR>1 && $1 ~ /GenericOp/ {print $20}' "$csv")
    n=$(echo "$vals" | wc -l)
    paste -d' ' <(tail -n "$n" "$here/run_order.txt") <(echo "$vals")
    cp "$csv" "$here/last_ops.csv"
    cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$here/last_profile_log_device.csv"
fi
rm -f "$marker"
