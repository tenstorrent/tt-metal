#!/bin/bash
# usage: r19_run.sh <ids comma> "<knobs>" [save_label]   (env: CHECK=1 to verify PCC)
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
export GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$2" GENIFY_MATMUL_CHECK="$CHECK"
FILE=tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r19.py
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile --run-all $FILE 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "$out" | grep -E "^PLAN|CHECK |FAILED|Error|passed|failed" | sed "s/epilogue=.*//" | head -30
all="res3_conv1,res4_conv1,res5_conv1,res5_short,dec_res2,dec_res3,aspp_1x1,aspp_proj,res3_short,res4_short"
ids=$(python -c "import sys; a=sys.argv[1].split(','); s=sys.argv[2].split(',') if sys.argv[2] else a; print(','.join(x for x in a if x in s))" "$all" "$1")
python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $(echo "$ids" | tr ',' ' ')
[ -n "$3" ] && mkdir -p /tmp/r19 && cp $(dirname $csv)/profile_log_device.csv /tmp/r19/$3.csv
