#!/usr/bin/env bash
# run_ric.sh <tag> <variants> [arms] [shape]: one profiled process over the combos, then medians.
# Log: perf_experiments/rm_issue_cost/logs/<tag>.log
set -u
ROOT=/localdev/pmilojevic/campaign/bh-lb-18/slots/chip3
cd "$ROOT"
export TT_METAL_HOME=$ROOT PYTHONPATH=$ROOT
source python_env/bin/activate
EXP=ttnn/ttnn/operations/g6_permute/perf_experiments/rm_issue_cost
T=tests/ttnn/unit_tests/operations/g6_permute/perf1/rm_issue_cost/test_rm_issue_cost.py
mkdir -p $EXP/logs
tag=$1
export RIC_VARIANTS=$2 RIC_ARMS=${3:-full,nogather} RIC_SHAPE=${4:-p44}
for attempt in 1 2 3; do
    TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --run-all --profile --no-py-zones "$T" > "$EXP/logs/$tag.log" 2>&1
    grep -q 'SAFE_PYTEST: PROFILER CSV' "$EXP/logs/$tag.log" && break
    echo "retry ($attempt) no CSV"
done
grep -a "SAFE_PYTEST_RESULT\|passed\|failed" "$EXP/logs/$tag.log" | tail -3
python3 $EXP/ric_medians.py "$EXP/logs/$tag.log" ${RIC_LAUNCHES:-12}
