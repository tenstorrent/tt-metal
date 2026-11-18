#!/bin/bash
# usage: r16_guard.sh <perf test file> <base knobs> [new knobs]  -> per-GenericOp device us, base vs new
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
run() {
  out=$(GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_KNOBS="$2" timeout 3000 scripts/run_safe_pytest.sh --profile "$1" 2>&1)
  echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //'
}
b=$(run "$1" "$2"); n=$(run "$1" "$3")
python3 - "$b" "$n" <<'PY'
import csv, sys
def load(p):
    return [r for r in csv.DictReader(open(p)) if r["OP CODE"].startswith("GenericOp")]
b, n = load(sys.argv[1]), load(sys.argv[2])
print(f"rows base {len(b)} new {len(n)}")
worst = 0
for i, (x, y) in enumerate(zip(b, n)):
    xb, yb = int(x["DEVICE KERNEL DURATION [ns]"]) / 1000, int(y["DEVICE KERNEL DURATION [ns]"]) / 1000
    d = (yb / xb - 1) * 100
    worst = max(worst, d)
    mark = " <<" if d > 3 else (" **" if d < -3 else "")
    print(f"{i:3d} cores {x['CORE COUNT']:>4s}->{y['CORE COUNT']:>4s} {xb:9.1f} {yb:9.1f} {d:+6.1f}%{mark}")
print(f"worst {worst:+.1f}%")
PY
