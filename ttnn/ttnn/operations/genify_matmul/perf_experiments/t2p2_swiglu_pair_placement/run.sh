#!/bin/bash
# usage: run.sh <ids> "<knobs>" [file] [save_label]
# env: GMP_PLACE / GMP_MAXW / GMP_FORCE_BANK / GMP_ALL (see place_plugin.py), CHECK=1, MAP=1 (rd_issue core map)
# Prints PLAN / GMPLAYOUT / CHECK lines and kernel us per op; appends a one-line record to runs.log.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_pair_placement
cd "$ROOT"
source python_env/bin/activate
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_TRACE=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$2" PYTHONPATH="$D:$PYTHONPATH"
[ -n "$CHECK" ] && export GENIFY_MATMUL_CHECK=1
FILE=${3:-tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r15.py}
LOG=$D/last_run.log
timeout 2400 scripts/run_safe_pytest.sh --run-all --profile -s -p place_plugin $FILE > $LOG 2>&1
rc=$?
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
grep -E "^PLAN|^TRACE|GMPLAYOUT|GMPGRAD|CHECK .*[0-9]\.[0-9]|FAILED|Error|error:|passed|failed|triage" $LOG | sed "s/epilogue=.*//" | cut -c1-300 | head -40
res=$([ -n "$csv" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $(echo "$1" | tr ',' ' '))
echo "$res"
echo "rc=$rc"
dev=$(dirname "$csv")/profile_log_device.csv
[ -n "$MAP" ] && [ -n "$csv" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/t2p2_coremap.py $dev rd_issue
[ -n "$4" ] && [ -n "$csv" ] && cp $dev /tmp/t2p2pp/$4.csv
echo "$(date +%T) ids=$1 knobs=$2 place=$GMP_PLACE maxw=$GMP_MAXW force=$GMP_FORCE_BANK file=$(basename $FILE) :: $(echo $res | tr '\n' ' ')" >> $D/runs.log
