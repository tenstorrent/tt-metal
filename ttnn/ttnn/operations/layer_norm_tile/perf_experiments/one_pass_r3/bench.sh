#!/bin/bash
# bench.sh <label> <variants> <cases> : one profiled pass of test_perf (variant-major, case-minor);
# prints DEVICE KERNEL DURATION (us) per GenericOp row, appends to logs/bench.txt, keeps the device log.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/one_pass_r3
marker=$(mktemp)
OP_VARIANTS="$2" OP_CASES="$3" scripts/run_safe_pytest.sh --run-all --profile "$here/test_one_pass_r3.py" -k test_perf --import-mode=prepend >"$here/logs/last_run.log" 2>&1
grep -E "passed|failed" "$here/logs/last_run.log" | tail -1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv"; exit 1; fi
rm -f "$marker"
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$here/logs/$1.device.csv"
python3 - "$csv" "$2" "$3" "$1" <<'PY' | tee -a "$here/logs/bench.txt"
import csv, sys
path, variants, cases, label = sys.argv[1:]
rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"]]
names = [f"{v}/{c}" for v in variants.split(",") for c in cases.split(",")]
print(f"== {label} ({len(rows)} rows, expected {len(names)})")
for name, r in zip(names, rows):
    print(f"  {name:28s} {int(r['DEVICE KERNEL DURATION [ns]'])/1000:7.2f} us")
PY
