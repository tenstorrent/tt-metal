#!/bin/bash
# usage: r17_run.sh "<only ids comma>" "<knobs>"  -> PLAN / CHECK lines + per-case device us
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$2"
FILE=tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r17.py
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile $FILE -s 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "$out" | grep -E "PLAN|CHECK|FAILED|Error|passed|skipped" | head -40
if [ -z "$1" ]; then names="qkvg lm_last in_proj out_proj down gate_up lm_head gate_up_b32 down_b32 out_proj_b16"; else names=$(echo "$1" | tr ',' ' '); fi
python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $names
echo "CSV $csv"
