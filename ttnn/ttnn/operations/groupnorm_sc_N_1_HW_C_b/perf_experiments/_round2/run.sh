#!/usr/bin/env bash
# Profiled bench.py run + parse. usage: R2_VARIANTS=... R2_CASES=... [R2_CHECK=0] run.sh <tag>
# Copies the device zone log next to the results (later runs overwrite generated/profiler).
set -u
REPO=/localdev/mmoscicki/gn-regen/tt-metal
HERE=$REPO/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/_round2
TAG=${1:-run}
mkdir -p $HERE/logs
LOG=$HERE/logs/$TAG.log
cd $REPO
source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --no-py-zones --run-all $HERE/bench.py --import-mode=prepend -s > $LOG 2>&1
echo "exit=$?  log=$LOG"
grep -E "SAFE_PYTEST_RESULT|SAFE_PYTEST: triage|AssertionError| passed| failed" $LOG | tail -4
CSV=$(grep -o "SAFE_PYTEST: PROFILER CSV: .*" $LOG | sed 's/SAFE_PYTEST: PROFILER CSV: //' | tail -1)
if [ -n "$CSV" ]; then
  cp $(dirname $CSV)/../../.logs/profile_log_device.csv $HERE/logs/$TAG.device.csv 2>/dev/null || cp $REPO/generated/profiler/.logs/profile_log_device.csv $HERE/logs/$TAG.device.csv
  python3 $HERE/parse.py "$CSV" --save $TAG | tee $HERE/logs/$TAG.txt
fi
