#!/bin/bash
# usage: GF_VARIANT=real|variant GF_MODE=N run.sh <tag> <-k expr>
cd "$(dirname "$0")/../../../../../.."
source python_env/bin/activate
D=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/gather_fast
mkdir -p $D/logs
tag=$1; shift
out=$(scripts/run_safe_pytest.sh --profile $D/test_gather_fast.py -k "$1" --import-mode=prepend 2>&1)
echo "$out" | grep -E "PRECISION|passed|failed|Error|error:|SAFE_PYTEST_RESULT|hang|triage" | grep -v "bind worker" | head -20
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*PROFILER CSV: //')
dev=$(dirname "$csv")/profile_log_device.csv
[ -f "$dev" ] || dev=generated/profiler/.logs/profile_log_device.csv
[ -f "$dev" ] && cp "$dev" $D/logs/$tag.csv && python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py $D/logs/$tag.csv --last | head -${ZL:-14} && python3 -I $D/startup.py $D/logs/$tag.csv
