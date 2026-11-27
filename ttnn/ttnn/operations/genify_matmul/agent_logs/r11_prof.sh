#!/bin/bash
# usage: KN="knobs" prof.sh <test file> <-k expr>
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
out=$(GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_KNOBS="$KN" timeout 1500 scripts/run_safe_pytest.sh --profile "$1" -k "$2" 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv
