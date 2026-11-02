#!/bin/bash
# Usage: TAIL_ORDER=base,x TAIL_CASES=8x512 [ZONES=1] run.sh   (profiler artifacts kept in tail_v2/prof)
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
export TT_METAL_HOME=$PWD
D=$PWD/ttnn/ttnn/operations/qkv_proj_heads_rope/perf_experiments/tail_v2
export TT_METAL_PROFILER_DIR=$D/prof
mkdir -p $D/logs
log=$D/logs/$(date +%H%M%S)_${TAIL_CASES:-8x512}_${TAIL_ORDER//,/_}.log
scripts/run_safe_pytest.sh --run-all $D/test_tail.py --import-mode=prepend -s > $log 2>&1
grep -E "^TAIL|passed|failed|Error|error:|^E  " $log | head -40
if [ -n "$ZONES" ]; then
  python3 $D/../_breakdown/zones.py $D/prof/.logs/profile_log_device.csv | grep -E "risc|KERNEL|matmul|last_k|store"
  python3 $D/zones_by_type.py $D/prof/.logs/profile_log_device.csv
fi
echo "LOG $log"
