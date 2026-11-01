#!/bin/bash
# Usage: whole.sh <variant> [pytest -k expr] -- profiles test_row (every perf + guard row), prints us per row.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/post_stats_chain
mkdir -p $here/results
tag=$1_$(date +%H%M%S)
marker=$(mktemp)
kexpr=${2:-}
LN_VARIANT=$1 scripts/run_safe_pytest.sh --run-all --profile "$here/test_whole_op.py::test_row" ${kexpr:+-k "$kexpr"} --import-mode=prepend -v >"$here/results/$tag.log" 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
if [ "$csv" -ot "$marker" ]; then echo "STALE CSV rc"; tail -5 "$here/results/$tag.log"; exit 1; fi
rm -f "$marker"
cp "$csv" "$here/results/$tag.ops.csv"
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$here/results/$tag.dev.csv"
ids=$(grep -oE "[0-9]+ (passed|failed)[^=]*" "$here/results/$tag.log" | tail -1)
ns=$(awk -F',' 'NR>1 && $1 ~ /GenericOp/ {printf "%.2f ", $20/1000}' "$csv")
echo "$1 | $ids"
echo "$1 | $ns"
