#!/bin/bash
# Usage: run.sh base-full,w2-comp,...   -> one pytest session under the profiler, then per-run summary.
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/wq_b_linear/perf_experiments/compute_dest_window
source python_env/bin/activate
CDW_RUNS="$1" scripts/run_safe_pytest.sh --profile --run-all $D/test_dest_window.py -s > $D/run.log 2>&1
grep -E "RESULT|passed|failed|FAILED|error:|SAFE_PYTEST: (RES|HANG|triage)" $D/run.log | grep -v "PCC nan"
csv=$(grep "SAFE_PYTEST: PROFILER CSV:" $D/run.log | awk '{print $NF}')
[ -n "$csv" ] && python3 $D/parse.py "$(dirname $csv)/profile_log_device.csv" "$1"
