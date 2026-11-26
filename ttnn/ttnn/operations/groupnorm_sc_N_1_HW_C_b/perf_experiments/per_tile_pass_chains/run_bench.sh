#!/usr/bin/env bash
# Run bench_variants.py under the device profiler and print per-test (case, variant, pcc, rms, device ns).
# usage: PTC_VARIANTS=base,cand [PTC_CASES=...] [PTC_ZONES=1] run_bench.sh <log_tag>
set -u
REPO=/localdev/mmoscicki/gn-regen/tt-metal
HERE=$REPO/ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/per_tile_pass_chains
TAG=${1:-run}
LOG=$HERE/logs/$TAG.log
mkdir -p $HERE/logs
cd $REPO
source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --no-py-zones $HERE/bench_variants.py --import-mode=prepend -s > $LOG 2>&1
echo "exit=$?  log=$LOG"
grep -E "SAFE_PYTEST_RESULT|SAFE_PYTEST: triage" $LOG | head -3
CSV=$(grep -o "SAFE_PYTEST: PROFILER CSV: .*" $LOG | sed 's/SAFE_PYTEST: PROFILER CSV: //' | tail -1)
python3 $HERE/parse_results.py "$LOG" "$CSV" | tee $HERE/logs/$TAG.txt
