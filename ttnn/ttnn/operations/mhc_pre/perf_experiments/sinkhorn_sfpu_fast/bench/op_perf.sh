#!/bin/bash
# usage: DEV=<card|auto> SKB_SHAPES=.. SKB_DTYPES=.. SKB_OPS=base,grad SKB_KNOBS=default SKB_REPEAT=5 [ZONES=1] bench/op_perf.sh
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
DEV=${DEV:-auto}
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
[ "$DEV" != auto ] && export TT_METAL_PROFILER_DIR="$PWD/generated/dev$DEV/profiler"
[ -n "$ZONES" ] && export MHC_PRE_KERNEL_DEFINES="KERNEL_PERF_ZONES"
scripts/run_safe_pytest.sh --device "$DEV" --run-all -s --timeout=3500 --import-mode=prepend \
  ttnn/ttnn/operations/mhc_pre/perf_experiments/sinkhorn_sfpu_fast/bench/test_op_perf.py 2>&1 \
  | grep -v 'riscv-tt-elf' | grep -E '^PERF |^BITWISE |^NONDET |^NANCHECK |passed|failed| error|Error|^E  |SAFE_PYTEST' | cut -c1-400
