#!/bin/bash
# usage: run_e2e.sh <label> <cases> <variants> [--dev]   e.g. run_e2e.sh r1 41 base,split
# Runs test_fns_e2e.py under the profiler; prints "label  fns<idx>|<variant>  median_ns  [3 calls]" (cmp_windows.py)
# and appends it to runs/e2e_ab.txt. Logs / ops CSV under runs/.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
label=$1; cases=$2; variants=$3; dev=$4
D=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fill_noc_split
mkdir -p $D/runs
out=$(FNS_CASES=$cases FNS_VARIANTS=$variants FNS_RESULTS=$D/runs/e2e_$label.jsonl timeout 3000 \
    scripts/run_safe_pytest.sh $dev --profile --run-all $D/test_fns_e2e.py --import-mode=prepend 2>&1)
echo "$out" > $D/runs/e2e_$label.log
echo "$out" | grep -E "FILL_NOC_SPLIT|passed|failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error:" | tail -30
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" $D/runs/e2e_${label}_ops.csv
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" \
    | awk -v l="$label" '{print l"\t"$0}' | tee -a $D/runs/e2e_ab.txt
