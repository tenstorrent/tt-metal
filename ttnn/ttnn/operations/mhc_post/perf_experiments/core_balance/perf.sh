#!/bin/bash
# usage: perf.sh <session> <variants> <shapes> [reps]   (device perf of the core_balance bake-off, then report)
set -o pipefail
REPO=$(cd "$(dirname "$0")"/../../../../../.. && pwd)
E=$REPO/ttnn/ttnn/operations/mhc_post/perf_experiments/core_balance
cd "$REPO" && source python_env/bin/activate
rm -f "$E/run_order_$1.jsonl"
OUT=$(CB_SESSION=$1 CB_VARIANTS=$2 CB_SHAPES=$3 CB_REPS=${4:-1} timeout 3000 scripts/run_safe_pytest.sh ${DEV:+--device $DEV} --profile --run-all \
  tests/ttnn/unit_tests/operations/mhc_post/perf_experiments_core_balance_test.py -k test_perf 2>&1 | tee /tmp/cb_perf_$1.log)
echo "$OUT" | grep -E "passed|failed|error|SAFE_PYTEST_RESULT|triage" | head
CARD=$(echo "$OUT" | grep -o "card(s)=[0-9]*" | head -1 | cut -d= -f2)
echo "card=$CARD"
cp "$REPO/generated/dev$CARD/profiler/.logs/cpp_device_perf_report.csv" "$E/perf_$1.csv"
cp "$REPO/generated/dev$CARD/profiler/.logs/profile_log_device.csv" "$E/zones_$1.csv" 2>/dev/null
python3 "$E/report.py" "$E/perf_$1.csv" "$1" | tee "$E/result_$1.txt"
