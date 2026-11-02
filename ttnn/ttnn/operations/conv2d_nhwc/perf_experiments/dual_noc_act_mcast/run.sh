#!/bin/bash
# usage: run.sh <tag> <impl> <case>   (env DNA_MODE / DNA_B16 / DNA_CHUNK pass through to the candidate)
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/dual_noc_act_mcast
mkdir -p $D/logs
tag=$1
out=$(scripts/run_safe_pytest.sh --profile "$D/test_dual_noc_act_mcast.py::test_dna[$2-$3]" --import-mode=prepend 2>&1)
echo "$out" | grep -E "PRECISION|passed|failed|Error|SAFE_PYTEST_RESULT|hang|triage" | grep -v "bind worker" | head -20
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*PROFILER CSV: //')
dev=$(dirname "$csv")/profile_log_device.csv
cp "$dev" $D/logs/$tag.csv 2>/dev/null && python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py $D/logs/$tag.csv --last | head -${ZL:-40}
