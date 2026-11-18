#!/bin/bash
# Usage: run.sh <case> <variants>      e.g. run.sh te_gate base,h=12-4
# Prints MS_RESULT lines and, per variant (in run order), the DEVICE KERNEL DURATION of genify_matmul.
# MS_ZONES=1 also prints the per-zone summary of the LAST variant. Profiler output -> this dir's prof/.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_msplit_b_resident
cd "$ROOT"
source python_env/bin/activate
export TT_METAL_PROFILER_DIR=$D/prof
rm -rf $D/prof
LOG=$D/last_run.log
GENIFY_MATMUL_PERF=1 MS_CASE=$1 MS_VARIANTS=$2 timeout 1800 scripts/run_safe_pytest.sh ${MS_DEV:+--dev} --profile -s tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/t2_msplit_b_resident/test_msplit.py > $LOG 2>&1
echo "exit=$?"
grep -E "MS_PLAN|MS_RESULT|passed|failed|error:|Error|SAFE_PYTEST_RESULT|triage|hang" $LOG | sed 's/ PLAN .*blk=/ blk=/' | sort -u | cut -c1-700
CSV=$(ls -t $D/prof/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -z "$CSV" ] && CSV=$(echo "$(grep 'PROFILER CSV' $LOG)" | sed 's/.*CSV: //')
if [ -n "$CSV" ] && [ -f "$CSV" ]; then
python3 - "$CSV" "$1" "$2" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("OP CODE", "").startswith("GenericOp")]
import os
cases = sys.argv[2].split(",")
vs = [f"{c}:{v}" for c in cases for v in sys.argv[3].split(",")]
rows = rows[-len(vs):]
for v, r in zip(vs, rows):
    print(f"MS_NS case={sys.argv[2]} variant={v} us={int(r['DEVICE KERNEL DURATION [ns]'])/1000:.1f} cores={r.get('CORE COUNT')}")
PY
LOGCSV=$(dirname $CSV)/../../.logs/profile_log_device.csv
[ -f "$D/prof/.logs/profile_log_device.csv" ] && LOGCSV=$D/prof/.logs/profile_log_device.csv
[ -n "$MS_ZONES" ] && python3 $D/zones.py $LOGCSV
echo "$(date +%T) $1 $2 $(grep MS_NS -c /dev/null)" >> $D/runs.log
else
  echo "no CSV"; tail -30 $LOG
fi
