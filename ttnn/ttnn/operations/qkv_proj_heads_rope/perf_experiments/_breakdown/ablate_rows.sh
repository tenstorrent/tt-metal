#!/bin/bash
# Usage: ablate_rows.sh <rows e.g. 4x512,2x512> <define,define,...|none>  -> device kernel ns per row (no correctness check)
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
rows=$1; defs=$2
if [ "$defs" != "none" ]; then export QKV_ABLATE=$defs; fi
log=/tmp/ablate_rows_$$.log
QKV_SWEEP=1 QKV_SWEEP_NOCHECK=1 QKV_SWEEP_ROWS=$rows QKV_SWEEP_HEADS=0 scripts/run_safe_pytest.sh --profile --run-all \
  tests/ttnn/unit_tests/operations/qkv_proj_heads_rope/test_qkv_proj_heads_rope_perf_sweep.py > $log 2>&1
grep -E "[0-9]+ (passed|failed)|SAFE_PYTEST: (WARNING|HANG)" $log | tail -2
csv=$(grep "PROFILER CSV:" $log | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$csv')) if 'Generic' in r['OP CODE']]
print('ABLATE', '$rows', '$defs', ' '.join(r['DEVICE KERNEL DURATION [ns]'] for r in rows))
"
[ -n "$ZONES" ] && python3 ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/_breakdown/zones.py
