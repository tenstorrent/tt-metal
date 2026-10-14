#!/bin/bash
# usage: bench.sh <shape_id> '<DUAL_KNOBS json>'  -> per-cluster span of each of 3 profiled samples + median
# (ZONES=1 also prints the per-zone table of the last sample)
cd "$(dirname "$0")/../../../../../.."
shape=$1; knobs=${2:-'{}'}
D=ttnn/ttnn/operations/tilize/perf_experiments/dual_noc_writes
log=/tmp/dnw_$$.log
DUAL_ITERS=3 DUAL_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile "$D/test_dual.py::test_dual[$shape]" > $log 2>&1
grep -q "1 passed" $log || { echo "$shape $knobs FAILED log=$log"; exit 1; }
csv=$(dirname $(grep "SAFE_PYTEST: PROFILER CSV:" $log | awk '{print $NF}'))/profile_log_device.csv
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('$csv')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)[-3:]))")
spans=""; for r in $runs; do spans="$spans $(python3 $T $csv --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
echo "$shape $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n 2p)"
[ -n "$ZONES" ] && python3 $T $csv --run $(echo $runs | awk '{print $NF}')
rm -f $log
