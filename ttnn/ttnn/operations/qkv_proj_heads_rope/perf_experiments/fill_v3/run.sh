#!/bin/bash
# Usage: run.sh <flags e.g. F3_LATESEED,F3_KSIZES=2:2:4:4:4:4:4 | base> <shapes e.g. 4x512,2x512>
# -> correctness + device kernel ns per shape. ZONES=1: fill timeline (single-shape runs only).
# DUMP=<tag>: save outputs to logs/<tag>_<shape>.pt (bit-identity check via cmp.py).
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
unset TT_METAL_DPRINT_CORES
D=ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/fill_v3
flags=$1; [ "$flags" = base ] && flags=""
export F3_FLAGS=$flags F3_SHAPES=$2
[ -n "$DUMP" ] && export F3_DUMP=$PWD/$D/logs/$DUMP
tag=$(echo "$1" | tr ',:=' '+._')
log=$D/logs/run_${tag}_${2//,/_}_$(date +%H%M%S).log
scripts/run_safe_pytest.sh --profile --run-all $PWD/$D/test_fill_v3.py --import-mode=prepend -s > $log 2>&1
grep -E "^FILL_V3 |passed|failed|SAFE_PYTEST_RESULT|SAFE_PYTEST: HANG|^E  .*(Assert|assert)" $log | head -20
csv=$(grep "PROFILER CSV:" $log | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$csv')) if 'Generic' in r['OP CODE']]
print('KERNEL_NS', '$1', '$2', ' '.join(r['DEVICE KERNEL DURATION [ns]'] for r in rows))
"
cp generated/profiler/.logs/profile_log_device.csv ${log%.log}_device.csv 2>/dev/null
[ -n "$ZONES" ] && python3 $D/first.py ${log%.log}_device.csv
[ -n "$OCC" ] && { python3 $D/occ.py ${log%.log}_device.csv TRISC_0 compute_kb_wait compute_bias_seed; python3 $D/occ.py ${log%.log}_device.csv BRISC writer_w_read writer_w_mcast_send writer_w_receive | grep -E "#[0-3] "; }
echo "LOG $log"
