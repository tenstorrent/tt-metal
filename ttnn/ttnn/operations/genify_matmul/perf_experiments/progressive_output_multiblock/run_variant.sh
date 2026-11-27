#!/bin/bash
# Usage: run_variant.sh <case> <variant> <mode>
# Prints PMB_RESULT (pcc) / plan / DEVICE KERNEL DURATION of genify_matmul and the zone summary.
# Profiler artifacts go to this dir's prof/ (TT_METAL_PROFILER_DIR) so sibling runs never overwrite them.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/progressive_output_multiblock
cd "$ROOT"
T=tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/progressive_output_multiblock/test_multiblock.py
export TT_METAL_PROFILER_DIR=$D/prof
rm -rf $D/prof
OUT=$(GENIFY_MATMUL_PERF=1 PMB_CASE=$1 PMB_VARIANT=$2 PMB_MODE=$3 scripts/run_safe_pytest.sh --profile -s $T 2>&1)
echo "$OUT" | grep -E "PMB_|passed|failed|Error|error:|SAFE_PYTEST_RESULT|triage" | grep -v "pcc=nan" | sed "s/ base=.*blk=/ blk=/" | sort -u | cut -c1-700
CSV=$(ls -t $D/prof/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
python3 - "$CSV" "$1" "$2" "$3" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    if "generic" in r.get("OP CODE", "").lower():
        print(f"PMB_NS case={sys.argv[2]} variant={sys.argv[3]} mode={sys.argv[4]} ns={r['DEVICE KERNEL DURATION [ns]']} cores={r.get('CORE COUNT')}")
PY
[ -n "$PMB_ZONES" ] && python3 $D/zones.py $D/prof/.logs/profile_log_device.csv
echo "$1 $2 $3 $(date +%T)" >> $D/runs.log
fi
