#!/bin/bash
cd /localdev/kgrujcic/tt-metal-codegen
for i in $(seq $2); do
  out="$1"
  for k in affineTILE 2x512 allL1_res; do
    scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py -k $k > /tmp/ln3/rep_run.log 2>&1
    csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    out="$out $k=$(awk -F',' 'NR>1 {printf "%.1f", $20/1000}' $csv)"
  done
  echo "$out"
done
