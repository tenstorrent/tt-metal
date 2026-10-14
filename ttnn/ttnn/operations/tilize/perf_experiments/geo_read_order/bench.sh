#!/bin/bash
# usage: bench.sh <shape_id> '<knobs json>'  -> median per-cluster span over GEO_ITERS (default 3) samples
# Reads this run's own profile_log_device.csv copy from its reports/<ts>/ dir (siblings share
# generated/profiler/.logs, which is overwritten by the next locked run).
cd "$(dirname "$0")/../../../../../.."
shape=$1; knobs=${2:-'{}'}; iters=${GEO_ITERS:-3}
D=ttnn/ttnn/operations/tilize/perf_experiments/geo_read_order
log=/tmp/geo_$$.log
GEO_ITERS=$iters GEO_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile "$D/test_geo.py::test_geo[$shape]" > $log 2>&1
grep -q "1 passed" $log || { echo "FAILED $shape $knobs"; grep -E "Error|assert|TT_THROW" $log | head -5; exit 1; }
csv=$(dirname "$(grep 'SAFE_PYTEST: PROFILER CSV:' $log | sed 's/.*PROFILER CSV: //')")/profile_log_device.csv
[ -f "$csv" ] || { echo "no device csv for $shape $knobs"; exit 1; }
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
runs=$(python3 -c "
import csv,sys;rows=list(csv.reader(open(sys.argv[1])))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)))" $csv)
spans=""; for r in $runs; do spans="$spans $(python3 $T $csv --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
n=$(echo $spans | wc -w)
echo "$shape $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n "$(( (n+1)/2 ))p")"
cp $csv /tmp/geo_last_profile.csv
[ -n "$PERCORE" ] && python3 $D/percore.py /tmp/geo_last_profile.csv
exit 0
