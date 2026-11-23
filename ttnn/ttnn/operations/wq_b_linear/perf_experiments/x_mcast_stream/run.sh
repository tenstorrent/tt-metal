#!/bin/bash
# Usage: run.sh "<XMS_KNOBS ;-separated>" [label] [--dev]  -> profiled bench + stage zones (span, x mcast zones, PCC)
cd "$(dirname "$0")"; D=$PWD; cd ../../../../../..
source python_env/bin/activate
if [ "$3" == "--dev" ]; then
  out=$(XMS_KNOBS="$1" scripts/run_safe_pytest.sh --dev $D/test_xms.py --rootdir=$D 2>&1)
  echo "== ${2:-$1} (dev)"; echo "$out" | grep -E "^PCC|passed|failed|Error|hang|triage" | head; exit
fi
out=$(XMS_KNOBS="$1" scripts/run_safe_pytest.sh --profile $D/test_xms.py --rootdir=$D 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
echo "== ${2:-$1}"
echo "$out" | grep -E "^PCC 0|passed|failed|RuntimeError|AssertionError|hang" | head -3
if [ -n "$csv" ]; then
  f="$(dirname $csv)/profile_log_device.csv"
  python3 $D/stages.py "$f" | grep -E "spans|x_mcast|x_read|BRISC .*x_wait|TRISC_2"
  echo "${2:-$1} csv=$f" >> $D/runs.log
else
  echo "$out" | tail -20
fi
