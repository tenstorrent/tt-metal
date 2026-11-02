#!/bin/bash
# Usage: run.sh <rows e.g. 8x512,4x512> <variants e.g. baseline,lag1>  -> device kernel ns per test (in order)
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
D=ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/epilogue_overlap
out=$(EO_ROWS=$1 EO_VARIANTS=$2 EO_HEADS=${3:-0} scripts/run_safe_pytest.sh --profile --run-all $D/test_epilogue_overlap.py --import-mode=prepend -s 2>&1)
echo "$out" | grep -E "EO_PREC|passed|failed|Error|error:|PROFILER CSV|hang" | grep -v "^\s*$" | tail -40
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 -c "
import csv,sys
rows=[r for r in csv.DictReader(open('$csv')) if 'Generic' in r['OP CODE'] or 'generic' in r['OP CODE'].lower()]
for r in rows: print('EO_NS', r['OP CODE'], r['DEVICE KERNEL DURATION [ns]'])
"
