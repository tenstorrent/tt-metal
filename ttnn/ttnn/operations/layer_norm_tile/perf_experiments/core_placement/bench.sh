#!/bin/bash
# Usage: bench.sh <variant> [pytest node suffix, default ::test_perf]
# Profiles one placement variant and prints one DEVICE KERNEL DURATION per GenericOp row.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/core_placement
node=${2:-::test_perf}
marker=$(mktemp)
LN_PLACEMENT=$1 scripts/run_safe_pytest.sh --run-all --profile "$here/test_placement.py$node" --import-mode=prepend >"$here/last_run.log" 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv"; exit 1; fi
rm -f "$marker"
echo "$1 $(awk -F',' 'NR>1 && $1 ~ /GenericOp/ {printf "%s ", $20}' "$csv")"
