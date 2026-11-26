#!/usr/bin/env bash
# Profiled bench.py run + parse. usage: CCP_VARIANTS=base,both [CCP_CASES=focus|sweep|a,b] run.sh <tag> [zones]
# Copies the device zone log right after the run (other runs overwrite it) when the 2nd arg is "zones".
set -u
REPO=/localdev/mmoscicki/gn-regen/tt-metal
HERE=$REPO/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/combine_critical_path
TAG=${1:-run}
LOG=$HERE/logs/$TAG.log
mkdir -p $HERE/logs
cd $REPO
source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --no-py-zones --run-all $HERE/bench.py --import-mode=prepend -s -k test_ccp > $LOG 2>&1
echo "exit=$?  log=$LOG"
grep -E "SAFE_PYTEST_RESULT|SAFE_PYTEST: triage|AssertionError|error:|passed|failed" $LOG | head -8
CSV=$(grep -o "SAFE_PYTEST: PROFILER CSV: .*" $LOG | sed 's/SAFE_PYTEST: PROFILER CSV: //' | tail -1)
ZARG=""
if [ "${2:-}" = "zones" ] && [ -n "$CSV" ]; then
  DEV=$(dirname $CSV)/profile_log_device.csv
  [ -f "$DEV" ] && ZARG="--zones $DEV"
fi
[ -n "$CSV" ] && python3 $HERE/parse.py "$CSV" $ZARG --save $TAG | tee $HERE/logs/$TAG.txt
