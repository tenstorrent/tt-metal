#!/bin/bash
# usage: run_flux.sh <label> <cases> [extra env assignments...]
# Runs the real-op FLUX harness (or a private copy via BRF_OP_COPY, see test_brf_op.py) profiled, prints medians.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
D=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/bank_run_fill
label=$1; cases=$2; shift 2; ENVS=("$@")
test=${BRF_TEST:-tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py}
out=$(env "${ENVS[@]}" FLUX_CMP_RESULTS=$PWD/$D/runs/$label.jsonl FLUX_CMP_CASES=$cases FLUX_CMP_SKIP_TTNN=1 timeout 2400 scripts/run_safe_pytest.sh --profile --run-all $test ${BRF_PYTEST_EXTRA} 2>&1)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error" | tail -5
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp "$csv" $D/runs/${label}_ops.csv
python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | awk -v l="$label" '{print l"\t"$1"\t"$2}' | tee -a $D/runs/results.txt
