#!/bin/bash
# Usage: LBF_FOLD=0|1 [LBF_ROWS=a,b] bench.sh tag
# Profiles the selected rows; prints "<tag> <row> <DEVICE KERNEL DURATION ns>" and the precision lines.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/lone_band_fold
tag=${1:-fold${LBF_FOLD:-1}}
log=$here/logs/$tag.log
mkdir -p $here/logs
PYTHONPATH=$here:${PYTHONPATH:-} scripts/run_safe_pytest.sh --run-all --profile "$here/test_lbf.py" -p lbf_swap --import-mode=prepend -s >"$log" 2>&1
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv $here/logs/$tag.device.csv 2>/dev/null
grep -E "passed|failed|FAILED|Error|SAFE_PYTEST_RESULT" "$log" | grep -v "^E " | tail -6
python3 $here/parse.py "$log" "$tag"
