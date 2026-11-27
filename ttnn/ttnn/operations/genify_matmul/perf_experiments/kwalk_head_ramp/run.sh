#!/bin/bash
# Usage: run.sh <case> <variants comma list> [timeline]
# One process: the variants in order (fresh program cache each), KH_RES lines (ns, pcc, diff vs base).
# With a 3rd arg (single variant): also the K-delivery timeline from this dir's private profiler dump.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/kwalk_head_ramp
cd "$ROOT"
source python_env/bin/activate
export TT_METAL_PROFILER_DIR=$D/prof
rm -rf $D/prof  # private profiler dump (not an artifact; delete after reading)
OUT=$(KH_CASE=$1 KH_VARIANTS=$2 scripts/run_safe_pytest.sh -s tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/kwalk_head_ramp/test_kh.py 2>&1)
echo "$OUT" | grep -E "KH |KH_RES|passed|failed|Error|error:|SAFE_PYTEST_RESULT|triage" | grep -v "^E " | cut -c1-400
echo "$OUT" | grep KH_RES | sed "s/^.*KH_RES/$(date +%T) KH_RES/" >> $D/results.log
if [ -n "$3" ]; then python3 $D/timeline.py $D/prof/.logs/profile_log_device.csv; fi
