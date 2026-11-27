#!/bin/bash
# usage: unit.sh <label> <pytest args...>   (env SS_SWAP=1 SS_DEFINES=GM_SS_STREAM): correctness net with the candidate
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_stream_epilogue
cd "$ROOT"; source python_env/bin/activate
LABEL=$1; shift
export PYTHONPATH="$D:$PYTHONPATH"
timeout 5400 scripts/run_safe_pytest.sh --run-all --precompile -p ss_plugin "$@" > $D/last_unit_$LABEL.log 2>&1
echo "exit=$?"
grep -E "SSPLAN" $D/last_unit_$LABEL.log | awk '{print $3,$4,$5,$6,$7,$8,$11}' | sort | uniq -c | sort -rn | head -30
grep -E "^FAILED|^ERROR| passed| failed|triage" $D/last_unit_$LABEL.log | tail -15
