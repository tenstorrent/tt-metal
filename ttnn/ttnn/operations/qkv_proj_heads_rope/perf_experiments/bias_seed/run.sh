#!/bin/bash
# Usage: run.sh <mode> <kb> <shapes e.g. 8x512,4x512> [heads]
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
export BS_MODE=$1 BS_KB=$2 BS_SHAPES=$3 BS_HEADS=${4:-0}
D=ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/bias_seed
log=$D/logs/run_$1_kb$2_${3//,/_}_h${4:-0}_$(date +%H%M%S).log
scripts/run_safe_pytest.sh --profile --run-all $PWD/$D/test_bias_seed.py --import-mode=prepend -s > $log 2>&1
grep -E "BIAS_SEED |passed|failed|Error|error:|^E " $log | grep -v "^E *$" | head -20
csv=$(grep "PROFILER CSV:" $log | sed 's/.*PROFILER CSV: //')
[ -n "$csv" ] && python3 $D/parse.py "$csv"
cp generated/profiler/.logs/profile_log_device.csv ${log%.log}_device.csv 2>/dev/null
[ -n "$ZONES" ] && python3 ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/_breakdown/zones.py ${log%.log}_device.csv
echo "LOG $log"
