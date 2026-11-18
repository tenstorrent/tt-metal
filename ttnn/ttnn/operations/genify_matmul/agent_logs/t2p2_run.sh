#!/bin/bash
# usage: run.sh <id> "<knobs>" [file] [save_label]
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$2" GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_CHECK="$CHECK"
FILE=${3:-tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r15.py}
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile $FILE 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "$out" | grep -E "^PLAN|CHECK |FAILED|Error|passed" | sed "s/epilogue=.*//" | head -20
python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $(echo "$1" | tr ',' ' ')
[ -n "$4" ] && cp $(dirname $csv)/profile_log_device.csv ${SAVE_DIR:-/tmp/t2p2}/$4.csv
