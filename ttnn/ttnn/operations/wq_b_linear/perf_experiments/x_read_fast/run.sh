#!/bin/bash
# Usage: run.sh "<XRD_KNOBS>" [label]  -> profiled bench + analysis (span, round 0 push, reads done, PCC)
cd "$(dirname "$0")"; D=$PWD; cd ../../../../../..
source python_env/bin/activate
out=$(XRD_KNOBS="$1" scripts/run_safe_pytest.sh --profile $D/test_x_read_fast.py --rootdir=$D 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "== ${2:-$1}"
echo "$out" | grep -E "^PCC 1|passed|failed|RuntimeError|AssertionError|error:" | head -5
[ -n "$csv" ] && echo "csv=$csv" && python3 $D/analyze.py "$(dirname $csv)/profile_log_device.csv" | tee -a $D/runs.log.tmp && echo "${2:-$1}" >> $D/runs.log.tmp
[ -z "$csv" ] && echo "$out" | tail -30
true
