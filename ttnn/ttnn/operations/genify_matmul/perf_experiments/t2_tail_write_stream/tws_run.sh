#!/bin/bash
# usage: tws_run.sh <suite: r16|r15|r11|flux|focus> "<only ids comma>" "<knobs>" "<TWS_DEFINES>"
#   env FORCE / BLOCK / CHECK=1 / TWS_REAL=1 / TWS_LAYOUT=1 / ZONES=1 / KFILTER=<pytest -k expr>
# Prints PLAN / CHECK lines and the DEVICE KERNEL DURATION per op (this dir's own profiler output dir).
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_tail_write_stream
cd "$ROOT"; source python_env/bin/activate
export PYTHONPATH=$D:$PYTHONPATH TT_METAL_PROFILER_DIR=$D/prof
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$2" GENIFY_MATMUL_KNOBS="$3" TWS_DEFINES="$4"
export TWS_GRAD="$GRAD" TWS_BITCHECK="$BITCHECK" TWS_POLICY="$POLICY" GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_BLOCK="$BLOCK" GENIFY_MATMUL_CHECK="$CHECK"
rm -rf $D/prof
F=tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_$1.py
K=(); [ -n "$KFILTER" ] && K=(-k "$KFILTER")
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile --run-all -s -p tws_plugin "${K[@]}" $F 2>&1)
echo "$out" > $D/last_run.log
echo "$out" | grep -E "^PLAN|^TWSBANK|^TILES|CHECK|LAYOUT|FAILED|error|Error|passed|failed|SAFE_PYTEST_RESULT|triage" | grep -v "Restored\|DEBUG" | sed "s/epilogue=.*//" | head -150
CSV=$(ls -t $D/prof/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && python3 - "$CSV" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    if "generic" in r.get("OP CODE", "").lower():
        print(f"NS {int(r['DEVICE KERNEL DURATION [ns]']):9d} cores={r.get('CORE COUNT')}")
PY
[ -n "$ZONES" ] && python3 $ROOT/ttnn/ttnn/operations/genify_matmul/agent_logs/t2p1_zone_summary.py $D/prof/.logs/profile_log_device.csv | grep -E "zone|wr_write|rd_tail|wr_wait_out|BRISC-KERNEL|NCRISC-KERNEL|cmp_matmul"
echo "$(date +%T) $1 [$2] [$3] [$4] F=$FORCE" >> $D/runs.log
