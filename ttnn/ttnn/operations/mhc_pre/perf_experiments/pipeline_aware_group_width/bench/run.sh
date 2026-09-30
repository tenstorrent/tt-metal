#!/bin/bash
# usage: GW_SHAPES=TxC,.. GW_VARIANTS=default,cand,w11,.. [GW_WDTYPE=bfloat16] [GW_REPEAT=3] [GW_CHECK=1] [DEV=N] run.sh [perf|check]
# perf  = TT_METAL_DEVICE_PROFILER on, DEVICE KERNEL DURATION per call; check = no profiler (use with GW_CHECK=1)
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
T=ttnn/ttnn/operations/mhc_pre/perf_experiments/pipeline_aware_group_width/bench/test_gw.py
DEVARG=""; [ -n "$DEV" ] && DEVARG="--device $DEV"
if [ "${1:-perf}" = perf ]; then
  D=${DEV:-auto}
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  [ -n "$DEV" ] && export TT_METAL_PROFILER_DIR="$PWD/generated/dev$DEV/profiler"
fi
scripts/run_safe_pytest.sh $DEVARG --run-all "$T" -s --timeout=3500 --import-mode=prepend 2>&1 \
  | grep -v 'riscv-tt-elf' | grep -E "^PERF |^CORRECT |^ROW |passed|failed| error|Error|^E  |SAFE_PYTEST" | cut -c1-500
