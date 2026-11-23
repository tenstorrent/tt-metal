#!/bin/bash
# Usage: run.sh "<XRF_KNOBS>" [label]  -> profiled bench + analysis (span, x round 0 arrival, PCC)
cd "$(dirname "$0")"; D=$PWD; cd ../../../../../..
source python_env/bin/activate
out=$(XRF_KNOBS="$1" scripts/run_safe_pytest.sh --profile $D/test_x_round0_fill.py --rootdir=$D 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "== ${2:-$1}"
echo "$out" | grep -E "^PCC 1|passed|failed|RuntimeError|AssertionError" | grep -v "PCC 0 nan" | head -3
[ -n "$csv" ] && python3 $D/analyze.py "$(dirname $csv)/profile_log_device.csv" && echo "${2:-$1} csv=$csv" >> $D/runs.log
