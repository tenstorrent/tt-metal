#!/bin/bash
# usage: run.sh <tag> <test function name>   (env knobs FRB_* pass through)
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/fewer_rounds_bigger_kblock
tag=$1
scripts/run_safe_pytest.sh --profile --run-all "$D/test_frb.py::$2" --import-mode=prepend > $D/logs/$tag.out 2>&1
grep -E "PLAN|PRECISION|L1_PEAK|passed|failed|Error:|SAFE_PYTEST_RESULT|hang|triage" $D/logs/$tag.out | grep -v "bind worker" | head -20
cp generated/profiler/.logs/profile_log_device.csv $D/logs/$tag.csv 2>/dev/null && python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py $D/logs/$tag.csv | grep -E "program|KERNEL (BRISC|TRISC_1)"
