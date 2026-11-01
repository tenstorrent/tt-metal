#!/bin/bash
# Usage: bench.sh <variant>  -- profiles the whole op (8 perf rows + 2 extras), prints DEVICE KERNEL DURATION per row.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/stats_fusion
marker=$(mktemp)
LN_STATS_VARIANT=$1 scripts/run_safe_pytest.sh --run-all --profile "$here/test_whole_op.py::test_perf" "$here/test_whole_op.py::test_extra" --import-mode=prepend >"$here/last_run.log" 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv"; exit 1; fi
rm -f "$marker"
echo "v$1 $(awk -F',' 'NR>1 && $1 ~ /GenericOp/ {printf "%s ", $20}' "$csv")  ($(grep -o 'SAFE_PYTEST_RESULT: [A-Z]*' $here/last_run.log))"
