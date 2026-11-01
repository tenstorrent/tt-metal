#!/bin/bash
# rep.sh <label> <n> [LN_KNOBS]: n profiled runs of the three focus rows; prints focus1/2/3 per run
cd /localdev/kgrujcic/tt-metal-codegen
for i in $(seq $2); do
  LN_KNOBS="$3" scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py -k "focus" > /tmp/ln3/rep_run.log 2>&1
  csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
  echo "$1 $(awk -F',' 'NR>1 {printf "%.1f ", $20/1000}' $csv)"
done
