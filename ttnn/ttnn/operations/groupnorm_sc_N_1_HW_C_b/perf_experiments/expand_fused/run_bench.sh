#!/usr/bin/env bash
# Profiled bench_expand.py run + parse. usage: EXPF_VARIANTS=v0,v1 [EXPF_CASES=..] [EXPF_AFFINE=..] run_bench.sh <tag> [zones]
set -u
REPO=/localdev/mmoscicki/gn-regen/tt-metal
HERE=$REPO/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/expand_fused
TAG=${1:-run}
ZONES=${2:-c_expand}
LOG=$HERE/logs/$TAG.log
mkdir -p $HERE/logs
cd $REPO
source python_env/bin/activate
export EXPF_TAG=$TAG
before=$(wc -l < $HERE/results.jsonl 2>/dev/null || echo 0)
scripts/run_safe_pytest.sh --profile --no-py-zones $HERE/bench_expand.py --import-mode=prepend -s > $LOG 2>&1
echo "exit=$?  log=$LOG"
grep -E "SAFE_PYTEST_RESULT|SAFE_PYTEST: triage|AssertionError|passed|failed|error" $LOG | head -5
after=$(wc -l < $HERE/results.jsonl 2>/dev/null || echo 0)
CSV=$(grep -o "SAFE_PYTEST: PROFILER CSV: .*" $LOG | sed 's/SAFE_PYTEST: PROFILER CSV: //' | tail -1)
[ -n "$CSV" ] && python3 $HERE/parse_run.py "$(dirname $CSV)" $((after - before)) "$ZONES" | tee $HERE/logs/$TAG.txt
