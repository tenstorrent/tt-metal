#!/bin/bash
# Usage: run.sh <case> <modes> [abl]   e.g. run.sh focus 0,1:50 epi
# Prints TWBN_ lines, and per variant (in run order) the DEVICE KERNEL DURATION of genify_matmul.
# TWBN_ZONES=1 also prints the zone summary of the LAST variant. Profiler output -> this dir's prof/.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/tail_writes_both_nocs
cd "$ROOT"
T=tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/tail_writes_both_nocs/test_tail_split.py
export TT_METAL_PROFILER_DIR=$D/prof
rm -rf $D/prof
LOG=$D/last_run.log
GENIFY_MATMUL_PERF=1 TWBN_CASE=$1 TWBN_MODES=$2 TWBN_ABL=$3 timeout 1500 scripts/run_safe_pytest.sh --profile -s $T > $LOG 2>&1
echo "exit=$?"
grep -E "TWBN_|passed|failed|error:|Error|SAFE_PYTEST_RESULT|triage" $LOG | grep -v "pcc=nan" | sort -u | cut -c1-600
CSV=$(ls -t $D/prof/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
python3 - "$CSV" "$1" "$2" "$3" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if "generic" in r.get("OP CODE", "").lower()]
modes = sys.argv[3].split(",")
rows = rows[-len(modes):]
for m, r in zip(modes, rows):
    print(f"TWBN_NS case={sys.argv[2]} variant={m} abl={sys.argv[4] or 'real'} ns={r['DEVICE KERNEL DURATION [ns]']} cores={r.get('CORE COUNT')}")
PY
[ -n "$TWBN_ZONES" ] && python3 $D/zones.py $D/prof/.logs/profile_log_device.csv
echo "$(date +%T) $1 $2 ${3:-real}" >> $D/runs.log
fi
