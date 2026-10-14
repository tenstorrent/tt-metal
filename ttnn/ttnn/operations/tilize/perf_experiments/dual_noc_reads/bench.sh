#!/bin/bash
# usage: bench.sh <shape_id> '<knobs json>'  -> median per-cluster span over 3 profiled samples
cd "$(dirname "$0")/../../../../../.."
shape=$1; knobs=${2:-'{}'}
D=ttnn/ttnn/operations/tilize/perf_experiments/dual_noc_reads
log=/tmp/dualnoc_$$.log
DUAL_NOC_ITERS=3 DUAL_NOC_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile "$D/test_dual_noc.py::test_dual_noc[$shape]" > $log 2>&1
grep -q "1 passed" $log || { tail -30 $log; exit 1; }
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('generated/profiler/.logs/profile_log_device.csv')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)))")
spans=""; for r in $runs; do spans="$spans $(python3 $T --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
echo "$shape $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n 2p)"
[ -n "$ZONES" ] && python3 $T
