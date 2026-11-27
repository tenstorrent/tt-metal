#!/bin/bash
# usage: run.sh "<case,...>" "<variant,...>" ["KNOB=val,..."] [save_label]
#   env: CHECK=1 (PCC / bit equality), ZONES=1 (per-zone stats), SAVE_DIR (default /tmp/t2p2swx) with save_label
# Prints SWXPLAN / SWXCHECK lines and, per GenericOp row (case-major, variant-minor), the device kernel us.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_bank_exchange
cd "$ROOT"
source python_env/bin/activate
export SWX_CASES="$1" SWX_VARIANTS="$2" SWX_KNOBS="$3" SWX_CHECK="$CHECK"
LOG=$D/last_run.log
timeout 2400 scripts/run_safe_pytest.sh --run-all --profile -s \
  tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/t2p2_swiglu_bank_exchange/test_swx_op.py > $LOG 2>&1
echo "exit=$?"
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
grep -E "^SWXPLAN|^SWXCHECK|^FAILED|^E  |passed|failed|SAFE_PYTEST: (triage|hang|TIMEOUT)" $LOG | grep -v "^SAFE_PYTEST: env" | sed "s/epilogue=.*//" | cut -c1-420 | head -60
IFS=',' read -ra CS <<< "$1"; IFS=',' read -ra VS <<< "$2"
names=""; for c in "${CS[@]}"; do for v in "${VS[@]}"; do names="$names $c/$v"; done; done
[ -n "$csv" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/r11_summary.py $csv $names
dev=$(dirname "$csv")/profile_log_device.csv
[ -n "$ZONES" ] && [ -n "$csv" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/t2p1_zone_summary.py $dev
[ -n "$4" ] && [ -n "$csv" ] && mkdir -p ${SAVE_DIR:-/tmp/t2p2swx} && cp $dev ${SAVE_DIR:-/tmp/t2p2swx}/$4.csv
echo "$(date +%T) cases=$1 variants=$2 knobs=$3" >> $D/runs.log
