#!/bin/bash
# usage: run.sh <label> <cases> <variants>    e.g. run.sh r1 41,47 base,reader
# Runs test_store_offload.py under the profiler and appends "label  so<idx>|<variant>  median_ns  [3 calls]" to
# /tmp/perf16/store_offload_ab.txt (and prints it). Extra env (SO_POLL_EVERY, ...) passes through.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
label=$1; cases=$2; variants=$3
D=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/store_offload
out=$(SO_CASES=$cases SO_VARIANTS=$variants SO_RESULTS=/tmp/perf16/so_$label.jsonl timeout 3000 \
    scripts/run_safe_pytest.sh --profile --run-all $D/test_store_offload.py --import-mode=prepend 2>&1)
echo "$out" > /tmp/perf16/so_${label}.log
echo "$out" | grep -E "STORE_OFFLOAD|passed|failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error:" | tail -40
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" /tmp/perf16/so_${label}_ops.csv
cp generated/profiler/.logs/profile_log_device.csv /tmp/perf16/so_${label}_dev.csv 2>/dev/null
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" \
    | awk -v l="$label" '{print l"\t"$0}' | tee -a /tmp/perf16/store_offload_ab.txt
