#!/bin/bash
# usage: prof.sh <variant real|wr> <tag> <-k expr> ; prints precision + per-RISC KERNEL max from THIS run's report dir
cd "$(dirname "$0")/../../../../../.." || exit 1
source python_env/bin/activate
L=ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/weights_on_reader_risc/logs
mkdir -p $L
out=$(WR_VARIANT=$1 scripts/run_safe_pytest.sh --profile ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/weights_on_reader_risc/test_wr.py -k "$3" --import-mode=prepend 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*PROFILER CSV: //')
echo "$out" | grep -E "PRECISION|SAFE_PYTEST_RESULT" | tr '\n' ' '
[ -n "$csv" ] || { echo "NO CSV"; exit 1; }
python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py "$(dirname $csv)/profile_log_device.csv" --last > $L/$2_$1.txt
echo; echo "$2 $1: $(grep 'KERNEL' $L/$2_$1.txt | tr -s ' ' | tr '\n' ' ')"
