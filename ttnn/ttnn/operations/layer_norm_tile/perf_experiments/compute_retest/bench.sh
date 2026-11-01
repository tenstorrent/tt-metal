#!/bin/bash
# Usage: bench.sh <variant> <pytest selector suffix, e.g. test_peel> -- profiles, prints DEVICE KERNEL DURATION per test.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/compute_retest
sel=${2:-test_peel}
tag=$1_${sel}_$(date +%H%M%S)
marker=$(mktemp)
LN_VARIANT=$1 scripts/run_safe_pytest.sh --run-all --profile "$here/test_retest.py::$sel" --import-mode=prepend >"$here/logs/$tag.log" 2>&1
rc=$?
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv rc=$rc"; tail -5 "$here/logs/$tag.log"; exit 1; fi
rm -f "$marker"
cp "$csv" "$here/logs/$tag.ops.csv"
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$here/logs/$tag.dev.csv" 2>/dev/null
grep -E "passed|failed" "$here/logs/$tag.log" | tail -1
echo "$1 $sel: $(awk -F',' 'NR>1 {printf "%.1f ", $20/1000}' "$csv")"
