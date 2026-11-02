#!/bin/bash
# Usage: perfrows.sh [test node]  -> correctness + DEVICE KERNEL DURATION [ns] per generic op, in test order
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
log=/tmp/perfrows_$$.log
scripts/run_safe_pytest.sh --profile --run-all "tests/ttnn/unit_tests/operations/qkv_proj_heads_rope/${1:-test_qkv_proj_heads_rope_perf.py::test_qkv_proj_heads_rope_perf}" > $log 2>&1
grep -E "[0-9]+ (passed|failed)|SAFE_PYTEST: WARNING" $log | tail -3
csv=$(grep "PROFILER CSV:" $log | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$csv')) if 'Generic' in r['OP CODE']]
print('KERNEL_NS', ' '.join(r['DEVICE KERNEL DURATION [ns]'] for r in rows))
"
