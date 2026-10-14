#!/bin/bash
# usage: bench.sh <shape_id> <variant csv or ""> ['<knobs json>']  -> median per-cluster span of 3 profiled runs
cd "$(dirname "$0")/../../../../../.."
shape=$1; var=$2; knobs=${3:-'{}'}
log=/tmp/rc_bench_$$.log
RC_ITERS=3 RC_VARIANT="$var" RC_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile \
  "ttnn/ttnn/operations/tilize/perf_experiments/reorder_cost/test_reorder_bench.py::test_reorder_bench[$shape]" > $log 2>&1
grep -q "PROFILER CSV" $log || { tail -40 $log; exit 1; }
grep -q "1 passed" $log || { grep -E "FAIL|Error|assert" $log | head; echo FAILED; }
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
csv=/tmp/rc_prof_$$.csv; cp generated/profiler/.logs/profile_log_device.csv $csv
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('$csv')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)))")
spans=""; for r in $runs; do spans="$spans $(python3 $T $csv --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
echo "$shape var=[$var] $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n 2p)"
python3 $T $csv 2>/dev/null | grep -E "reorder|read_barrier|issue|cb_reserve|NCRISC-KERNEL" | head -12
