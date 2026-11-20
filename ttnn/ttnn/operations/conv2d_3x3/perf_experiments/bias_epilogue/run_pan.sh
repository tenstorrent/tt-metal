#!/bin/bash
# usage: run_pan.sh <label> <case indices> <variants>   e.g. run_pan.sh pan1 0,1,2 base,epi
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/bias_epilogue
label=$1; cases=$2; variants=$3; shift 3
env BE_LOG=$label BE_PAN_CASES=$cases BE_PAN_VARIANTS=$variants "$@" $E/safe_run.sh $E/runs/$label.log --profile --run-all $E/test_panoptic_be.py; [ $? -eq 75 ] && exit 75; out=$(cat $E/runs/$label.log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage" | cut -c1-300 | tail -10
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" $E/runs/${label}_ops.csv
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
