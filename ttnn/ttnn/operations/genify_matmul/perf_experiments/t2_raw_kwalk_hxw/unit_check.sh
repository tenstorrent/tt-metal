#!/bin/bash
# Correctness gate: the op's own unit tests with this dir's compute kernel (HXW_SWAP=1) + the graduation defines.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw
cd "$ROOT"; source python_env/bin/activate
export PYTHONPATH="$D:$PYTHONPATH" HXW_SWAP=1 HXW_DEFINES="${HXW_DEFINES-GM_HXW+GM_HXW_STREAM}"
U=tests/ttnn/unit_tests/operations/genify_matmul
FILES=$(ls $U/test_genify_matmul*.py)
LOG=$D/last_unit.log
timeout 7000 scripts/run_safe_pytest.sh --run-all --precompile -p hxw_plugin $FILES "$@" > $LOG 2>&1
echo "exit=$?"
grep -E "passed|failed" $LOG | tail -3; grep -E "^FAILED|^ERROR" $LOG | head -20
grep "^HXWPLAN" $LOG | awk '{print $3}' | sort | uniq -c | sort -rn | head -20
