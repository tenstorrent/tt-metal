#!/bin/bash
# Usage: [OCI_ORDER=reverse] bench.sh <tag> [pytest -k expr]  -- profiled isolated bench; prints id + kernel ns.
here=$(cd "$(dirname "$0")" && pwd)
tag=$1; shift
out=$here/results/$tag; mkdir -p "$out"
cd /localdev/kgrujcic/tt-metal-codegen || exit 1
source python_env/bin/activate
marker=$(mktemp); sleep 1
k=()
[ -n "$1" ] && k=(-k "$1")
timeout 2400 scripts/run_safe_pytest.sh --profile --run-all "$here/test_bench.py" "${k[@]}" --import-mode=prepend -s > "$out/pytest.log" 2>&1
prof=/localdev/kgrujcic/tt-metal/generated/profiler
csv=$(find $prof/reports -name 'ops_perf_results_*.csv' -newer "$marker" | sort | tail -1)
rm -f "$marker"
grep -E "SAFE_PYTEST_RESULT|passed|failed" "$out/pytest.log" | tail -3
[ -z "$csv" ] && { echo "no CSV"; grep -E "TT_THROW|error|hang" "$out/pytest.log" | head; exit 1; }
cp "$csv" "$out/ops.csv"
grep -oE "OCI_BENCH variant=[a-z0-9_]+ rows=[0-9]+ Wt=[0-9]+ reps=[0-9]+ fp32_dest=[A-Za-z]+" "$out/pytest.log" | sed "s/OCI_BENCH //; s/ /,/g" > "$out/ids.txt"
awk -F',' 'NR>1 && $1 ~ /GenericOp/ {print $20}' "$out/ops.csv" > "$out/ns.txt"
python3 "$here/table.py" "$out"
