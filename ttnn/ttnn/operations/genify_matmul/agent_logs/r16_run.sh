#!/bin/bash
# usage: r16_run.sh "<only ids comma>" "<knobs>"   (FORCE="gm,gn[,placement]" env to pin a split)
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$2" GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_BLOCK="$BLOCK"
FILE=${3:-tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r16.py}
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile $FILE 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "$out" | grep -E "PLAN|CHECK|FAILED|Error|SKIPPED|passed|skipped" | sed "s/epilogue=.*//" | head -40
names=$(echo "$1" | tr ',' ' ')
python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $names
echo "CSV $csv"
