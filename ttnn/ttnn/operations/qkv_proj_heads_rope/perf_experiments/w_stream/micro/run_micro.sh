#!/bin/bash
# Usage: run_micro.sh [case,case]  -> DEVICE KERNEL DURATION [ns] per case in order
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
D=ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/w_stream/micro
[ -n "$1" ] && export MICRO_CASES=$1
log=$D/../logs/micro_$(date +%H%M%S).log
scripts/run_safe_pytest.sh --profile --run-all $PWD/$D/test_micro.py --import-mode=prepend -s > $log 2>&1
grep -E "^MICRO |passed|failed|Error" $log | head
csv=$(grep "PROFILER CSV:" $log | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$csv')) if 'Generic' in r['OP CODE']]
print('KERNEL_NS', ' '.join(r['DEVICE KERNEL DURATION [ns]'] for r in rows))
"
echo LOG $log
