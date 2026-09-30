#!/bin/bash
# usage: DEV=1 [DT=xbf16_wf32] [V=base,a,b,ab] [R=5] [ZONES=1] perf.sh "640x7168,640x1792,1280x4096"
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
DEV=${DEV:-1}
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_PROFILER_DIR="$PWD/generated/dev$DEV/profiler"
export PRELUDE_SHAPES="$1" PRELUDE_DTYPES=${DT:-xbf16_wf32} PRELUDE_VARIANTS=${V:-base,a,b,ab} PRELUDE_REPEAT=${R:-5}
[ -n "$ZONES" ] && export MHC_PRE_KERNEL_DEFINES="KERNEL_PERF_ZONES"
scripts/run_safe_pytest.sh --device "$DEV" --run-all -s --timeout=0 \
  ttnn/ttnn/operations/mhc_pre/perf_experiments/prelude_off_critical_path/test_prelude.py::test_perf 2>&1 \
  | grep -v 'riscv-tt-elf' | grep -E '^PERF .*us \||passed|failed| error|Error:|SAFE_PYTEST_RESULT' | cut -c1-400
