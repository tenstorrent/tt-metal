#!/bin/bash
# Correctness gate: the op's own unit tests + golden suite (default -k swiglu) with this dir's descriptor + kernels.
# usage: unit_check.sh [extra pytest args]   (env K: -k expression, default "swiglu")
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_bank_exchange
cd "$ROOT"; source python_env/bin/activate
export PYTHONPATH="$D:$PYTHONPATH"
U=tests/ttnn/unit_tests/operations/genify_matmul
LOG=$D/last_unit.log
timeout 7000 scripts/run_safe_pytest.sh --run-all --precompile -p swx_plugin $U/test_genify_matmul*.py \
  eval/golden_tests/genify_matmul/test_golden.py -k "${K-swiglu}" "$@" > $LOG 2>&1
echo "exit=$?"
grep -E "passed|failed" $LOG | tail -3; grep -E "^FAILED|^ERROR" $LOG | head -20
grep "^SWXPLAN" $LOG | awk '{print $3, $4, $5}' | sort | uniq -c | sort -rn | head -20
