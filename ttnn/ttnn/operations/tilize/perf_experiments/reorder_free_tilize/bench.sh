#!/bin/bash
# usage: bench.sh <test id e.g. 1x1x16384x64-bf16> <RFT 0|1> ['<knobs json>']
#   -> per-cluster span (zones.py) of each of 3 profiled samples + median; zone means of the last.
cd "$(dirname "$0")/../../../../../.."
id=$1; rft=$2; knobs=${3:-'{}'}
log=/tmp/rft_bench_$$.log
RFT_ITERS=3 RFT=$rft RFT_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile \
  "ttnn/ttnn/operations/tilize/perf_experiments/reorder_free_tilize/test_rft_bench.py::test_rft_bench[$id]" > $log 2>&1
grep -q "PROFILER CSV" $log || { tail -40 $log; exit 1; }
grep -q "1 passed" $log || { grep -E "FAIL|Error|assert" $log | head; echo FAILED; }
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
csv=/tmp/rft_prof_$$.csv; cp generated/profiler/.logs/profile_log_device.csv $csv
# runs: first = reference real op (skip), then RFT_ITERS samples
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('$csv')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)[1:]))")
spans=""; for r in $runs; do spans="$spans $(python3 $T $csv --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
echo "$id rft=$rft $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n 2p)"
python3 $T $csv 2>/dev/null | grep -E "reorder|read_barrier|reader_issue|compute_tilize|writer_cb_wait|KERNEL" | head -12
