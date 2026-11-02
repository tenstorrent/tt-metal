#!/bin/bash
# Usage: run.sh <row e.g. 8x512> <real|fork> [ramp e.g. 2,4]  -> correctness + device kernel ns
# Env passthrough: QKV_FS_WMIN (99 = force R1), ZONES=1 (zone table).
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
unset TT_METAL_DPRINT_CORES
export QKV_FS_ROWS=$1 QKV_FS_VARIANT=$2 QKV_FILL_RAMP=${3:-}
log=$(mktemp)
scripts/run_safe_pytest.sh --profile -s \
  ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/fill_shrink/test_fill_shrink.py --import-mode=prepend >$log 2>&1
grep -E "FILL_SHRINK|passed|failed|Error|SAFE_PYTEST_RESULT|hang" $log | grep -v "riscv-tt-elf\|nan" | tail -8
csv=$(grep -o "PROFILER CSV: .*" $log | awk '{print $3}')
python3 -c "
import csv
for x in csv.DictReader(open('$csv')): print('RESULT', '$1', '$2', 'ramp=${3:-none}', 'wmin=${QKV_FS_WMIN:-}', 'lag=${QKV_FS_TABLE_LAG:-0}', 'xl0=${QKV_FS_XLOCAL0:-0}', 'kernel_ns', x['DEVICE KERNEL DURATION [ns]'])
"
[ -n "$ZONES" ] && python3 ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/_breakdown/zones.py "$(dirname $csv)/profile_log_device.csv" | head -60
[ -n "$ROUNDS" ] && python3 ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/fill_shrink/rounds.py "$(dirname $csv)/profile_log_device.csv" $ROUNDS
rm -f $log
true
