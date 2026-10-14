#!/bin/bash
# usage: bench.sh <shape_id> '<knobs json>' [iters]  -> per-sample per-cluster spans + median; GRID=1 prints per-core map
cd "$(dirname "$0")/../../../../../.."
shape=$1; knobs=${2:-'{}'}; iters=${3:-3}
D=ttnn/ttnn/operations/tilize/perf_experiments/geo_write_noc
log=/tmp/geo_$$.log
export TT_METAL_PROFILER_DIR=/tmp/geo_write_noc_prof; rm -rf $TT_METAL_PROFILER_DIR; mkdir -p $TT_METAL_PROFILER_DIR
CSV=$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv
GEO_ITERS=$iters GEO_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile "$D/test_geo.py::test_geo[$shape]" > $log 2>&1
grep -q "1 passed" $log || { tail -30 $log; exit 1; }
[ -f $CSV ] || { echo "no csv at $CSV"; ls -R $TT_METAL_PROFILER_DIR | head; exit 1; }
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('$CSV')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)))")
spans=""; for r in $runs; do spans="$spans $(python3 $T $CSV --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
n=$(echo $spans | wc -w); mid=$(( (n+1)/2 ))
echo "$shape $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n ${mid}p)"
[ -n "$GRID" ] && python3 $D/grid.py writer_flush $CSV
[ -n "$ZONES" ] && python3 $T $CSV
exit 0
