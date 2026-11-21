#!/bin/bash
# usage: ab.sh <label> <tokens> <cases>   (tokens: comma list of fill,weights,store,tilize,bias,matmul; "" = full)
# Needs the hooks (apply_hooks.py) in the op for any token. Window medians go to $OUT/ab_results.txt.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
OUT=${OUT:-/tmp/perf20}
mkdir -p $OUT
label=$1; toks=$2; cases=$3
out=$(CONV2D_3X3_ABLATE=$toks FLUX_CMP_RESULTS=$OUT/ab_$label.jsonl FLUX_CMP_CASES=$cases FLUX_CMP_SKIP_TTNN=1 ${EXTRA_ENV} timeout 570 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py 2>&1)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage" | tail -3
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp "$csv" $OUT/ab_${label}_ops.csv
cp generated/profiler/.logs/profile_log_device.csv $OUT/ab_${label}_dev.csv
python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | awk -v l="$label" '{print l"\t"$1"\t"$2}' | tee -a $OUT/ab_results.txt
