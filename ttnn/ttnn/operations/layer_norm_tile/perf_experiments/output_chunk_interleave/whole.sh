#!/bin/bash
# Usage: whole.sh <variant> <tag> [-k expr]  -- profiled whole-op rows; prints row + DEVICE KERNEL DURATION + precision.
here=$(cd "$(dirname "$0")" && pwd)
variant=$1; tag=$2; shift 2
out=$here/results/whole_$tag; mkdir -p "$out"
cd /localdev/kgrujcic/tt-metal-codegen || exit 1
source python_env/bin/activate
marker=$(mktemp); sleep 1
LN_OCI=$variant timeout 2400 scripts/run_safe_pytest.sh --profile --run-all "$here/test_whole_op.py" "$@" --import-mode=prepend -s > "$out/pytest.log" 2>&1
prof=/localdev/kgrujcic/tt-metal/generated/profiler
csv=$(find $prof/reports -name 'ops_perf_results_*.csv' -newer "$marker" | sort | tail -1)
rm -f "$marker"
grep -E "SAFE_PYTEST_RESULT|passed|failed" "$out/pytest.log" | tail -2
[ -z "$csv" ] && { echo "no CSV"; grep -E "TT_THROW|error|hang" "$out/pytest.log" | head; exit 1; }
cp "$csv" "$out/ops.csv"
cp "$prof/.logs/profile_log_device.csv" "$out/" 2>/dev/null
grep -oE "test_row\[[^]]+\]" "$out/pytest.log" | awk '!seen[$0]++' | sed 's/test_row\[//; s/\]//' > "$out/ids.txt"
grep -oE "OCI_PREC .*" "$out/pytest.log" | awk '{print $3, $4}' | tail -n "$(wc -l < "$out/ids.txt")" > "$out/prec.txt"
awk -F',' 'NR>1 && $1 ~ /GenericOp/ {print $20}' "$out/ops.csv" > "$out/ns.txt"
paste "$out/ids.txt" "$out/ns.txt" "$out/prec.txt"
