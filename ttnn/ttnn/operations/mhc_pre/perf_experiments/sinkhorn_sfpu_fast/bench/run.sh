#!/bin/bash
# usage: [SKB_VARIANTS=0,1,..] [SKB_ITERS=20] [SKB_REPS=11] [DEV=auto] bench/run.sh correct|perf|slot_map [--dev]
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
T=ttnn/ttnn/operations/mhc_pre/perf_experiments/sinkhorn_sfpu_fast/bench/test_sinkhorn_bench.py
MODE=$1; shift
DEV=${DEV:-auto}
if [ "$MODE" = perf ]; then
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
fi
scripts/run_safe_pytest.sh --device "$DEV" "$@" --run-all "$T" -s --import-mode=prepend --timeout=3500 -k "test_$MODE" 2>&1 \
  | grep -v 'riscv-tt-elf' | grep -E "^PERF |^CORRECT |passed|failed| error|Error|^E  |SAFE_PYTEST|assert" | cut -c1-600
