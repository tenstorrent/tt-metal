#!/bin/bash
# Usage: run.sh <flags e.g. XFIRST,BMC | base> <shapes e.g. 8x512,4x512>  -> correctness + device kernel ns per shape
# ZONES=1: zone table (single-shape runs only).
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
unset TT_METAL_DPRINT_CORES
D=ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/w_stream
flags=$1; [ "$flags" = base ] && flags=""
export WS_FLAGS=$flags WS_SHAPES=$2
[ -n "$FORK" ] && export WS_FORK=$FORK
log=$D/logs/run_${1//,/+}_${2//,/_}_$(date +%H%M%S).log
scripts/run_safe_pytest.sh --profile --run-all $PWD/$D/test_w_stream.py --import-mode=prepend -s > $log 2>&1
grep -E "^W_STREAM |passed|failed|SAFE_PYTEST_RESULT|^E  .*(Assert|assert)" $log | grep -v "pcc=nan" | head -20
csv=$(grep "PROFILER CSV:" $log | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$csv')) if 'Generic' in r['OP CODE']]
print('KERNEL_NS', '$1', ' '.join(r['DEVICE KERNEL DURATION [ns]'] for r in rows))
"
cp generated/profiler/.logs/profile_log_device.csv ${log%.log}_device.csv 2>/dev/null
python3 $D/span.py ${log%.log}_device.csv
[ -n "$ZONES" ] && python3 $D/../_breakdown/zones.py ${log%.log}_device.csv
echo "LOG $log"
