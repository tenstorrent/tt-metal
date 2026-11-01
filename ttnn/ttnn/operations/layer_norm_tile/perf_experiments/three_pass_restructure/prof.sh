#!/bin/bash
# Usage: LN3P_MODE=<m> [LN_ABLATE=read,write] ./prof.sh <tag> [pytest node suffix]
# Profiled run; saves ops_perf CSV + device log under results/<tag>/ and prints per-case kernel ns.
here=$(cd "$(dirname "$0")" && pwd)
tag=$1; shift
out=$here/results/$tag; mkdir -p "$out"
cd /localdev/kgrujcic/tt-metal-codegen || exit 1
source python_env/bin/activate
marker=$(mktemp); sleep 1
timeout 1500 scripts/run_safe_pytest.sh --profile --run-all \
  "ttnn/ttnn/operations/layer_norm_tile/perf_experiments/three_pass_restructure/test_3p.py$1" \
  --import-mode=prepend -s > "$out/pytest.log" 2>&1
prof=/localdev/kgrujcic/tt-metal/generated/profiler
csv=$(find $prof/reports -name 'ops_perf_results_*.csv' -newer "$marker" | sort | tail -1)
cp "$prof/.logs/profile_log_device.csv" "$out/" 2>/dev/null
rm -f "$marker"
[ -z "$csv" ] && { echo "no CSV"; grep -E "TT_THROW|error:|hang" "$out/pytest.log" | head; exit 1; }
cp "$csv" "$out/ops.csv"
grep -oE "test_3p.py::[a-z_]+\[[^]]+\]" "$out/pytest.log" | awk '!seen[$0]++' | sed 's/.*\[//; s/\]//' > "$out/ids.txt"
awk -F',' 'NR>1 {print $20}' "$out/ops.csv" > "$out/ns.txt"
paste "$out/ids.txt" "$out/ns.txt"
