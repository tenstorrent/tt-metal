#!/bin/bash
# usage: guard_pan.sh <label> <cases>   Panoptic compare cases on the op currently in place; medians to /tmp/perf19/guard/pan_<label>.txt
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
L=$1; O=/tmp/perf19/guard; mkdir -p $O
out=$(CONV_CMP_CASES=$2 CONV_CMP_RESULTS=$O/pan_$L.jsonl timeout 570 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_panoptic_conv3x3_compare.py 2>&1)
echo "$out" | grep -E "^=+ .*(passed|failed)|SAFE_PYTEST_RESULT|HANG" | tail -2
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | awk -F'\t' -v l="$L" '{print l"\t"$1"\t"$2}' >> $O/pan_$L.txt
