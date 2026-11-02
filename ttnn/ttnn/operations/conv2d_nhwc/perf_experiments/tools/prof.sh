#!/bin/bash
# prof.sh <test_path> <-k expr> : one profiled run; prints pass/fail and per-core max KERNEL cycles.
cd "$(dirname "$0")/../../../../../.." || exit 1
source python_env/bin/activate
out=$(scripts/run_safe_pytest.sh --profile "$1" -k "$2" 2>&1)
d=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //; s#/ops_perf.*##')
res=$(echo "$out" | grep -oE "[0-9]+ passed|[0-9]+ failed|PRECISION[^\"]*" | sort -u | tr '\n' ' ')
echo "== $2 :: $res"
python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py "$d/profile_log_device.csv" | grep -E "KERNEL TRISC_1|program"
