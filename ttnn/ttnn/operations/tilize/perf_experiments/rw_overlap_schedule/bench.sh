#!/bin/bash
# usage: bench.sh <shape_id e.g. 1x1x16384x64> '<knobs json>' [zones]  -> median span of 3 profiled samples
cd "$(dirname "$0")/../../../../../.."
shape=$1; knobs=${2:-'{}'}
D=ttnn/ttnn/operations/tilize/perf_experiments/rw_overlap_schedule
log=/tmp/rwbench_$$.log
RW_ITERS=3 RW_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile "$D/test_bench.py::test_bench[$shape]" > $log 2>&1
grep -q "PROFILER CSV" $log || { tail -30 $log; exit 1; }
grep -q "1 passed" $log || { echo "CORRECTNESS FAIL"; grep -E "Error|assert|failed" $log | head; exit 1; }
cp generated/profiler/.logs/profile_log_device.csv /tmp/rwbench_$$.csv
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('/tmp/rwbench_$$.csv')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)))")
spans=""; for r in $runs; do spans="$spans $(python3 $T /tmp/rwbench_$$.csv --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
med=$(echo $spans | tr ' ' '\n' | sort -n | sed -n 2p)
echo "$shape $knobs PASS spans:$spans median: $med"
[ "$3" = zones ] && python3 $T /tmp/rwbench_$$.csv | grep -v KERNEL
rm -f $log /tmp/rwbench_$$.csv
