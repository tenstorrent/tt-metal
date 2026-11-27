#!/bin/bash
# usage: r17_sweep.sh "<only ids comma>" "<C list comma>" "<knobs>" -> device us per (case, C) in run order
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_CSWEEP="$2" GENIFY_MATMUL_KNOBS="$3"
FILE=tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r17.py
out=$(timeout 3000 scripts/run_safe_pytest.sh --profile $FILE -s 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "$out" | grep -E "FAILED|Error" | head -5
# the profiled (second) pass prints each PLAN once more; keep the last occurrence count of labels
names=$(echo "$out" | grep "^PLAN" | awk '{print $2"_"$7"_"$8}' | tail -n $(python -c "import csv,sys;print(sum(1 for r in csv.DictReader(open('$csv')) if r['OP CODE'].startswith('GenericOp')))"))
python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $names
echo "CSV $csv"
