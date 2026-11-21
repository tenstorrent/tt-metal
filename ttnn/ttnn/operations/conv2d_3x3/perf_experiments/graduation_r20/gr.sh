#!/bin/bash
# usage: gr.sh <label> <cases> [variants]   (graduation_r20 harness; window medians -> /tmp/perf20/gr_results.txt)
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
label=$1; cases=$2; vars=${3:-base,cand}
out=$(GR_LOG=$label GR_CASES=$cases GR_VARIANTS=$vars GR_ITERS=${GR_ITERS:-3} GR_ORDER=${GR_ORDER:-alt} timeout 580 scripts/run_safe_pytest.sh --profile --run-all ttnn/ttnn/operations/conv2d_3x3/perf_experiments/graduation_r20/test_final.py 2>&1)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|Error" | tail -4
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp "$csv" /tmp/perf20/gr_${label}_ops.csv
python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | awk -v l="$label" '{print l"\t"$0}' | tee -a /tmp/perf20/gr_results.txt
rm -rf "$(dirname "$csv")"
