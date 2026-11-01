#!/bin/bash
# bench.sh <variant base|half> <label> [rows]: one profiled pass; per-row device kernel us -> logs/<label>.txt
cd /localdev/kgrujcic/tt-metal-codegen
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/lone_band_half_tile
log=/tmp/lbh/$2.log
PYTHONPATH="$PWD/$here:${PYTHONPATH:-}" LBH_VARIANT=$1 LBH_ROWS="$3" scripts/run_safe_pytest.sh --run-all --profile \
  "$here/test_lbh.py" -p lbh_swap --import-mode=prepend -s >"$log" 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
python3 - "$csv" "$log" "$1" "$2" <<'PY' | tee -a $here/logs/bench.txt
import csv, re, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if "enericOp" in r.get("OP CODE", "") or True]
names = list(dict.fromkeys(m.group(1) for m in re.finditer(r"LBH_PRECISION \S+ (\S+) ", open(sys.argv[2]).read())))
prec = {m.group(1): m.group(2) for m in re.finditer(r"LBH_PRECISION \S+ (\S+) (pcc=\S+ rel_rms=\S+)", open(sys.argv[2]).read())}
durs = [int(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if r["DEVICE KERNEL DURATION [ns]"]]
print(f"[{sys.argv[3]}] {sys.argv[4]} csv={sys.argv[1]}")
for n, d in zip(names, durs):
    print(f"  {n:32s} {d/1000:7.2f} us  {prec.get(n,'')}")
if len(names) != len(durs):
    print("  WARN rows/durations mismatch", len(names), len(durs), durs)
PY
grep -E "[0-9]+ (passed|failed)" "$log" | tail -1
