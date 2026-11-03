#!/bin/bash
# Usage: MK_VARIANTS=... MK_SHAPE=... run.sh <label> : one fresh profiled run, per-program KERNEL cycles.
cd "$(dirname "$0")/../../../../../../.." || exit 1
D=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/matmul_kblock_cost/bench
export TT_METAL_PROFILER_DIR="$PWD/$D/prof/$1"
rm -rf "$TT_METAL_PROFILER_DIR"; mkdir -p "$TT_METAL_PROFILER_DIR"
source python_env/bin/activate
scripts/run_safe_pytest.sh --run-all --profile $D/test_kblock_bench.py --import-mode=prepend > "$TT_METAL_PROFILER_DIR/pytest.log" 2>&1
grep -E "^MK |FAILED|Timeout|Error|passed|failed" "$TT_METAL_PROFILER_DIR/pytest.log" | grep -v warning | cut -c1-200 | head -30
csv=$(find "$TT_METAL_PROFILER_DIR" -name profile_log_device.csv | head -1)
[ -n "$csv" ] && python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py "$csv" | grep -E "program|KERNEL"
