#!/bin/bash
# usage: [CBP_SHAPES=..] [CBP_VARIANTS=..] [CBP_DTYPES=..] [CBP_REPEAT=3] run.sh correct|perf [--dev]
# Card 2 only (other agents own cards 0 / 1).
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
T=ttnn/ttnn/operations/mhc_pre/perf_experiments/cross_block_pipeline/bench/test_cbp.py
MODE=$1; shift
if [ "$MODE" = perf ]; then
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  export TT_METAL_PROFILER_DIR="$PWD/generated/dev2/profiler"
fi
scripts/run_safe_pytest.sh --device 2 "$@" --run-all "$T" -s --timeout=3500 --import-mode=prepend -k "test_$MODE" 2>&1 \
  | grep -v 'riscv-tt-elf' | grep -E "^PERF |^CORRECT |^FAIL |passed|failed| error|Error|^E  |SAFE_PYTEST" | cut -c1-600
