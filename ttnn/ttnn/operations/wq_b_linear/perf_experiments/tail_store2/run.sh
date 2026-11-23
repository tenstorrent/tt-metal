#!/bin/bash
# usage (from anywhere): run.sh <label> "<WQB_KNOBS>"
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/wq_b_linear/perf_experiments/tail_store2
source python_env/bin/activate
out=$(WQB_KNOBS="$2" scripts/run_safe_pytest.sh --profile $D/test_tail2.py -s 2>&1)
echo "$out" | grep -E "^PCC|passed|failed| error:|HANG|triage" | grep -v "bind worker\|warning" | head -20
csv=$(echo "$out" | grep "SAFE_PYTEST: PROFILER CSV:" | awk '{print $NF}')
[ -n "$csv" ] && { echo "== $1 [$2]"; python3 $D/tail.py "$(dirname $csv)/profile_log_device.csv"; } | tee -a $D/results.log
