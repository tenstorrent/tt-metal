#!/bin/bash
# usage: run_profile.sh <variant> <tag>  -- one traced test_perf_ring profile run of a variant; prints the 6-shape summary
# and keeps the ops CSV path + device log under logs/.
R=/tmp/replay_hawoelwa/clone
E=$R/ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/early_next_load
cd $R; v=$1; tag=${2:-$1}
EARLY_VARIANT=$v timeout 3000 true --profile "tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_experiments/early_next_load/test_early_next_load.py::test_perf_ring" > $E/logs/perf_$tag.log 2>&1
csv=$(grep -o "PROFILER CSV: .*" $E/logs/perf_$tag.log | sed 's/PROFILER CSV: //')
echo "== $tag ($v) $(grep -E '[0-9]+ (passed|failed)' $E/logs/perf_$tag.log | tail -1)"
echo "$csv" > $E/logs/csv_$tag.txt
python $E/../_tools/traced_summary.py $csv
cp generated/profiler/.logs/profile_log_device.csv $E/logs/device_$tag.csv
