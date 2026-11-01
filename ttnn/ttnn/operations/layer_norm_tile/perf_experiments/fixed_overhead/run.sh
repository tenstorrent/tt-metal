#!/bin/bash
# usage: run.sh <nodeid>...  (relative to the layer_norm_tile test dir); env LN_ABLATE / LN_KNOBS pass through
cd /localdev/kgrujcic/tt-metal-codegen && source python_env/bin/activate
d=tests/ttnn/unit_tests/operations/layer_norm_tile
ids=(); for x in "$@"; do ids+=("$d/$x"); done
out=$(scripts/run_safe_pytest.sh --run-all --profile "${ids[@]}" 2>&1)
csv=$(echo "$out" | grep "OPs csv generated at" | sed "s/.*generated at: //")
[ -z "$csv" ] && { echo "$out" | tail -20; exit 1; }
echo "$csv" > /tmp/ln_r5/last_csv
python3 - "$csv" <<'PY'
import csv,sys
for r in csv.DictReader(open(sys.argv[1])):
    print(r['OP CODE'][:30], r['DEVICE KERNEL DURATION [ns]'], 'cores', r.get('CORE COUNT'))
PY
