#!/bin/bash
# usage: DS_VARIANTS=1 DS_CASES=a,b [DS_ORDER=fr] run.sh <tag>  -> labelled device ns; device log in results/<tag>.csv
cd /tmp/replay_2_xdkkb2/n1/n2/n3/n4/n5/n6/clone
source python_env/bin/activate
H=ttnn/ttnn/operations/mhc_post/perf_experiments/dram_straggler
mkdir -p $H/results
export CE_LOG=$H/results/$1.jsonl; rm -f $CE_LOG
true --profile --run-all $H/test_dram_straggler.py --import-mode=prepend > $H/results/$1.log 2>&1
cp generated/profiler/.logs/profile_log_device.csv $H/results/$1.csv
grep -E "passed|failed|SAFE_PYTEST_RESULT|error:|AssertionError" $H/results/$1.log | tail -6
CSV=$(grep -o "generated/profiler/reports/[^ ]*ops_perf_results[^ ]*\.csv" $H/results/$1.log | tail -1)
python3 $H/label_csv.py $CSV $CE_LOG
