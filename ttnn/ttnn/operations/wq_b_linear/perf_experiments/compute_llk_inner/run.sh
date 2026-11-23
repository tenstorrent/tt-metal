#!/bin/bash
# Usage: run.sh "<WQB_KNOBS>"  -> runs the bench under the profiler and prints PCC + zone summary.
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/wq_b_linear/perf_experiments/compute_llk_inner
source python_env/bin/activate
out=$(WQB_KNOBS="$1" scripts/run_safe_pytest.sh --profile $D/test_bench.py -s 2>&1)
echo "$out" | grep -v warning | grep -E "PCC|passed|failed|error:|SAFE_PYTEST: (PROFILER|triage|HANG)" | head -20
csv=$(echo "$out" | grep "SAFE_PYTEST: PROFILER CSV:" | awk '{print $NF}')
[ -n "$csv" ] && python3 $D/stages.py "$(dirname $csv)/profile_log_device.csv"
