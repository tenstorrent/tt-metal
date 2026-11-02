#!/bin/bash
# Usage: HF_IMPL=... HF_DEFINES=... HF_SET=... HF_CASE=... run.sh <label>
# One fresh profiled run of test_handshake_floor.py; profiler artifacts in ./prof/<label>, summary printed.
cd "$(dirname "$0")/../../../../../.." || exit 1
D=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/handshake_floor
export TT_METAL_PROFILER_DIR="$PWD/$D/prof/$1"
rm -rf "$TT_METAL_PROFILER_DIR"; mkdir -p "$TT_METAL_PROFILER_DIR"
source python_env/bin/activate
scripts/run_safe_pytest.sh --profile $D/test_handshake_floor.py --import-mode=prepend > "$TT_METAL_PROFILER_DIR/pytest.log" 2>&1
grep -E "^HF |PRECISION|FAILED|Timeout|assert|Error" "$TT_METAL_PROFILER_DIR/pytest.log" | grep -v "warning" | cut -c1-300 | sort -u | head -8
csv=$(find "$TT_METAL_PROFILER_DIR" -name profile_log_device.csv | head -1)
echo "== $1 ($csv)"
[ -n "$csv" ] && python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py "$csv" --last | grep -E "KERNEL|markers|^  [a-z_]+ +BRISC|^ +BRISC"
