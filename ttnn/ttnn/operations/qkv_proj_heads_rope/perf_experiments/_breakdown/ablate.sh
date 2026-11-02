#!/bin/bash
# Usage: ablate.sh <row e.g. 8x512> <define,define,...|none>  -> prints device kernel ns + zone table
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
unset TT_METAL_DPRINT_CORES
row=$1; defs=$2
b=${row%x*}; s=${row#*x}
if [ "$defs" != "none" ]; then export QKV_ABLATE=$defs; fi
QKV_SWEEP=1 QKV_SWEEP_NOCHECK=1 QKV_SWEEP_ROWS=$row QKV_SWEEP_HEADS=0 scripts/run_safe_pytest.sh --profile \
  tests/ttnn/unit_tests/operations/qkv_proj_heads_rope/test_qkv_proj_heads_rope_perf_sweep.py 2>&1 | grep -E "passed|failed|error:" | tail -3
csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
python3 -c "
import csv
for x in csv.DictReader(open('$csv')): print('ABLATE', '$row', '$defs', 'kernel_ns', x['DEVICE KERNEL DURATION [ns]'])
"
[ -n "$ZONES" ] && python3 ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/_breakdown/zones.py
