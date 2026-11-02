#!/bin/bash
# usage: TOF_IMPL=.. TOF_CASE=.. TOF_DEFINES=.. run.sh <tag>
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tilize_own_fast
mkdir -p $D/logs
out=$(scripts/run_safe_pytest.sh --profile $D/test_tilize_own_fast.py --import-mode=prepend 2>&1)
echo "$out" | grep -E "PRECISION|passed|failed|Error|error:|SAFE_PYTEST_RESULT|hang|triage|PROFILER CSV" | grep -v "bind worker" | head -20
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*PROFILER CSV: //')
dev=$(dirname "$csv")/profile_log_device.csv
[ -f "$dev" ] && cp "$dev" $D/logs/$1.csv && python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py $D/logs/$1.csv --last | head -${ZL:-30}
