#!/bin/bash
# usage: run.sh <case ids comma> "<knobs>" [test file]   env: FORCE, BLOCK, NSPLIT_KDIR=1 (kernel copy), ZONES=1, CHECK=1
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_progressive_nsplit
cd $ROOT
source python_env/bin/activate
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$2" GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_BLOCK="$BLOCK"
[ -n "$CHECK" ] && export GENIFY_MATMUL_CHECK=1
export TT_METAL_PROFILER_DIR=$D/prof PYTHONPATH=$D:$PYTHONPATH
rm -rf $D/prof
FILE=${3:-tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r16.py}
MODEFLAG=--profile; [ -n "$DEV" ] && MODEFLAG=--dev
out=$(timeout 2400 scripts/run_safe_pytest.sh $MODEFLAG $FILE -s -p nsplit_plugin 2>&1)
echo "$out" > $D/last_run.log
echo "$out" | grep -E "NSPLIT|PLAN|CHECK|FAILED|Error|error:|passed|failed|SAFE_PYTEST_RESULT|triage" | sed "s/epilogue=.*//" | sort -u | head -30
csv=$(ls -t $D/prof/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$csv" ] && python $ROOT/ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $(echo "$1" | tr ',' ' ')
[ -n "$ZONES" ] && python $ROOT/ttnn/ttnn/operations/genify_matmul/agent_logs/t2p1_zone_summary.py $D/prof/.logs/profile_log_device.csv
echo "$(date +%T) $1 knobs=[$2] FORCE=$FORCE BLOCK=$BLOCK KDIR=$NSPLIT_KDIR $(grep -h kernel_us <(python $ROOT/ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $(echo "$1" | tr ',' ' ') 2>/dev/null) | tr '\n' ' ')" >> $D/runs.log
