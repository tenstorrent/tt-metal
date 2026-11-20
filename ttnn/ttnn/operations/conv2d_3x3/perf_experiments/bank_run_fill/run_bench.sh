#!/bin/bash
# usage: run_bench.sh <label> [env assignments...]   (e.g. BRF_CASES=41 BRF_VARIANTS=base,bank)
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
D=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/bank_run_fill
label=$1; shift
out=$(env "$@" timeout 2400 scripts/run_safe_pytest.sh --profile --run-all $D/test_brf.py -k perf --import-mode=prepend 2>&1)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|Error" | tail -4
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
mkdir -p $D/runs && cp "$csv" $D/runs/bench_${label}_ops.csv
python3 $D/report_brf.py "$csv" | awk -v l="$label" '{print l"\t"$0}' | tee -a $D/runs/bench_results.txt
