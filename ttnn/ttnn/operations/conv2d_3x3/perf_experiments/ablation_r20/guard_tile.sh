#!/bin/bash
# usage: guard_tile.sh <label> <cases>   FLUX single-device (1x1 group, TILE input) cases; medians to /tmp/perf20/guard/tile_<label>.txt
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
L=$1; O=/tmp/perf20/guard; mkdir -p $O
out=$(FLUX_CMP_CASES=$2 FLUX_CMP_SKIP_TTNN=1 FLUX_CMP_RESULTS=$O/tile_$L.jsonl timeout 570 scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py 2>&1)
echo "$out" | grep -E "^=+ .*(passed|failed)|SAFE_PYTEST_RESULT|HANG" | tail -2
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | awk -F'\t' -v l="$L" '{print l"\t"$1"\t"$2}' >> $O/tile_$L.txt
