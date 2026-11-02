#!/bin/bash
# Usage: run.sh <variants e.g. base,kb8> <cases e.g. 8x512> [iso]   (ZONES=1 -> zone table of the last variant)
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
unset TT_METAL_DPRINT_CORES
D=ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/matmul_kloop
export MMK_VARIANTS=$1 MMK_CASES=$2
[ "$3" = "iso" ] && export MMK_ISO=1
log=$D/logs/run_${1//,/_}_${2//,/_}_${3:-e2e}_$(date +%H%M%S).log
scripts/run_safe_pytest.sh --profile --run-all -s $PWD/$D/test_mmk.py --import-mode=prepend > $log 2>&1
grep -E "^MMK |passed|failed|Error|error:|^E |SAFE_PYTEST|hang" $log | grep -v "^E *$" | head -40
cp generated/profiler/.logs/profile_log_device.csv ${log%.log}_device.csv 2>/dev/null
[ -n "$ZONES" ] && python3 ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/_breakdown/zones.py ${log%.log}_device.csv | grep -i "trisc\|zone"
echo "LOG $log"
