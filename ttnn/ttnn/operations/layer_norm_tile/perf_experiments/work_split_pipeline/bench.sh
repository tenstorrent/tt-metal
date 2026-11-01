#!/bin/bash
# Usage: WS_SPLIT=<strategy> [WS_OUT_DEPTH=n] [WS_ROWS=a,b] bench.sh [tag]
# Profiles the selected rows under one split strategy; prints "<tag> <row> <DEVICE KERNEL DURATION ns>".
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/work_split_pipeline
tag=${1:-${WS_SPLIT:-base}}
log=$here/logs/$tag.log
mkdir -p $here/logs
PYTHONPATH=$here:${PYTHONPATH:-} scripts/run_safe_pytest.sh --run-all --profile "$here/test_ws.py" -p ws_swap --import-mode=prepend >"$log" 2>&1
grep -E "passed|failed|FAILED|Error|SAFE_PYTEST_RESULT" "$log" | grep -v "^E " | tail -8
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv $here/logs/$tag.device.csv 2>/dev/null
names=(focus1 focus2 focus3 allL1_res L1_nores 8x384 32x32 affTILE g_bf16dest g_3x500 g_64x17 g_2x512)
[ -n "${WS_ROWS:-}" ] && IFS=, read -ra names <<<"$WS_ROWS"
mapfile -t durs < <(grep -oE "(PASSED|FAILED).*perf summary \(device=0\): count=1, min=[0-9]+" "$log" | sed -E 's/^(PASSED|FAILED).*min=([0-9]+)/\1 \2/')
for idx in "${!names[@]}"; do echo "$tag ${names[$idx]} ${durs[$idx]:-MISSING}"; done
