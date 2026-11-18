#!/bin/bash
# usage: run.sh <id[,id..]> "<KNOB=val,...>" [label]
# env: ASR_SWAP=1 (writer from ./kernels), ASR_DEFINES='A+B' (extra defines), ABL=1 (A alone: compute + B stubbed),
#      CHECK=1 (PCC), ZONES=1 (zone summary), FILE=<perf test file> (default r15)
# Bank config knobs for the focus: BANK_LOCAL_SWIGLU=True,BANK_LOCAL_ROW_FILL=False
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_act_stream_rounds
cd "$ROOT"
source python_env/bin/activate
KN="$2"
[ -n "$ABL" ] && KN="${KN:+$KN,}ABLATE_DEFINES='SKIP_COMPUTE+CKL_ELTWISE_CHAIN_SKIP_COMPUTE+GM_ABLATE_B'"
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$1" GENIFY_MATMUL_KNOBS="$KN" PYTHONPATH="$D:$PYTHONPATH"
[ -n "$CHECK" ] && export GENIFY_MATMUL_CHECK=1
FILE=${FILE:-tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r15.py}
LOG=$D/last_run.log
timeout 2400 scripts/run_safe_pytest.sh --run-all --profile -s -p asr_plugin $FILE > $LOG 2>&1
echo "exit=$?"
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
grep -E "^PLAN|ASRPLAN|ASRHASH|CHECK |FAILED|error:|passed|failed|triage" $LOG | sed "s/epilogue=.*//" | cut -c1-330 | head -30
[ -n "$csv" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $(echo "$1" | tr ',' ' ')
dev=$(dirname "$csv")/profile_log_device.csv
[ -n "$ZONES" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/t2p1_zone_summary.py $dev | grep -E "risc|wr_a|wr_wait|wr_final|BRISC.*kernel"
[ -n "$3" ] && cp $dev /tmp/t2p2asr/$3.csv
echo "$(date +%T) ids=$1 knobs=$KN swap=$ASR_SWAP defs=$ASR_DEFINES abl=$ABL label=$3 $(grep -o 'kernel_us=.*' <(python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv 2>/dev/null) | tr '\n' ' ')" >> $D/runs.log
