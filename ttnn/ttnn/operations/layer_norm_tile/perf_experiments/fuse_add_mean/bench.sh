#!/bin/bash
# Usage: FAM_VARIANTS=0,1 FAM_CASES=focus1 [FAM_PRELOAD=1] bench.sh <tag>
# Profiles the selected (variant x case) grid; prints one DEVICE KERNEL DURATION per GenericOp row
# (order = variant-major, case-minor) and keeps the device log for zones.py.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/fuse_add_mean
marker=$(mktemp)
scripts/run_safe_pytest.sh --run-all --profile "$here/test_fuse_add_mean.py" --import-mode=prepend >"$here/last_run.log" 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv"; exit 1; fi
rm -f "$marker"
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$here/last_profile_log_device.csv"
echo "$1 V=${FAM_VARIANTS:-all} C=${FAM_CASES:-all} P=${FAM_PRELOAD:-0}: $(awk -F',' 'NR>1 && $1 ~ /GenericOp/ {printf "%s ", $20}' "$csv")"
