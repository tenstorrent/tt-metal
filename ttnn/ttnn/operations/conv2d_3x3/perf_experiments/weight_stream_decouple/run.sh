#!/bin/bash
# usage: run.sh <label> <cases> <variants> [extra env...]   e.g. run.sh diag1 43,47 base,diag
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_stream_decouple
label=$1; cases=$2; variants=$3; shift 3
out=$(env WSD_LOG=$label WSD_CASES=$cases WSD_VARIANTS=$variants "$@" timeout 3000 scripts/run_safe_pytest.sh --profile --run-all $E/test_wsd.py 2>&1)
echo "$out" > $E/runs/$label.log
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|WSD:" | cut -c1-400 | tail -30
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" $E/runs/${label}_ops.csv
[ -f generated/profiler/.logs/profile_log_device.csv ] && gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
