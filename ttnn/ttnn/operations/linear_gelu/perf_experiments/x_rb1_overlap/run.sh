#!/bin/bash
# Usage: X1_VARIANTS=baseline,i1fe X1_CASES=8x512 run.sh  -> "<variant> <case> pcc bitwise <ns>" per op, in run order.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
cd "$REPO" && source python_env/bin/activate
mkdir -p "$HERE/_logs"
tag=$(date +%H%M%S)
scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_x_rb1.py" -s -p no:randomly --import-mode=prepend >"$HERE/_logs/$tag.log" 2>&1
grep -E "passed|failed" "$HERE/_logs/$tag.log" | tail -1
# Siblings share the profiler output dir: take this run's report path from its own log.
csv=$(grep -o "/localdev[^ ]*reports/[^ ]*ops_perf_results_[^ ]*\.csv" "$HERE/_logs/$tag.log" | tail -1)
cp "$(dirname "$csv")/profile_log_device.csv" "$HERE/_logs/$tag.zones.csv"
paste -d' ' <(grep -o "X1_RESULT.*" "$HERE/_logs/$tag.log" | grep -v "pcc=nan") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv") \
    | tee -a "$HERE/results.log"
echo "log $HERE/_logs/$tag.log"
