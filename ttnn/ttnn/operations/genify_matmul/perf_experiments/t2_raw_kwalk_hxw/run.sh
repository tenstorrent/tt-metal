#!/bin/bash
# usage: run.sh "<file:case,...>" "<variants>" [knobs]   env: FORCE, BLOCK, CHECK=1, ABL=1 (compute-only), ZONES=1, FP32=0
# Prints HXW lines (equality / PCC) and per GenericOp row (in run order: case-major, variant-minor) kernel_us.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw
cd "$ROOT"
source python_env/bin/activate
KN="$3"
[ -n "$ABL" ] && KN="${KN:+$KN,}ABLATE_DEFINES='GM_ABLATE_A+GM_ABLATE_A_SEND+GM_ABLATE_B'"
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_KNOBS="$KN" GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_BLOCK="$BLOCK"
export HXW_CASES="$1" HXW_VARIANTS="$2" HXW_FP32="${FP32:-1}"
[ -n "$CHECK" ] && export GENIFY_MATMUL_CHECK=1
LOG=$D/last_run.log
timeout 3000 scripts/run_safe_pytest.sh --run-all --profile -s tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw/test_hxw_op.py::test_hxw_op > $LOG 2>&1
echo "exit=$?"
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
grep -E "^PLAN|HXW |FAILED|Error|error:|passed|failed|triage" $LOG | sed "s/epilogue=.*//" | cut -c1-400 | head -60
IFS=',' read -ra CS <<< "$1"; IFS=',' read -ra VS <<< "$2"
names=""; for c in "${CS[@]}"; do for v in "${VS[@]}"; do names="$names ${c#*:}/$v"; done; done
[ -n "$csv" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $names
[ -n "$ZONES" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/t2p1_zone_summary.py generated/profiler/.logs/profile_log_device.csv
echo "$(date +%T) cases=$1 variants=$2 knobs=$KN force=$FORCE abl=$ABL" >> $D/runs.log
