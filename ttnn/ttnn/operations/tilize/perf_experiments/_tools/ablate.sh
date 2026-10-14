#!/bin/bash
# usage: ablate.sh <shape_id> '<knobs json>'   -> prints median span over 3 profiled samples + per-zone table of last run
shape=$1; knobs=${2:-'{}'}
TILIZE_PERF_ITERS=3 TILIZE_PERF_KNOBS="$knobs" scripts/run_safe_pytest.sh --profile "tests/ttnn/unit_tests/operations/tilize/test_tilize_perf.py::test_tilize_perf[$shape]" > /tmp/ablate_$$.log 2>&1
grep -q "PROFILER CSV" /tmp/ablate_$$.log || { tail -30 /tmp/ablate_$$.log; exit 1; }
T=ttnn/ttnn/operations/tilize/perf_experiments/_tools/zones.py
runs=$(python3 -c "
import csv;rows=list(csv.reader(open('generated/profiler/.logs/profile_log_device.csv')))[2:];print(' '.join(sorted({r[7] for r in rows},key=int)))")
spans=""; for r in $runs; do spans="$spans $(python3 $T --run $r 2>/dev/null | head -1 | sed 's/.*= \([0-9]*\) ns.*/\1/')"; done
echo "$shape $knobs spans:$spans median: $(echo $spans | tr ' ' '\n' | sort -n | sed -n 2p)"
