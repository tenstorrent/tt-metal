#!/bin/bash
# Tournament-2 Perf 1 guard: run one perf file (all cases, --profile) and save the ops CSV path under a label.
# usage: t2p1_guard.sh <label> <perf test file> ; t2p1_guard.sh cmp <file stem>  -> base vs new per GenericOp row
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
if [ "$1" == "cmp" ]; then
python3 - "/tmp/t2p1_guard_base_$2.csv" "/tmp/t2p1_guard_new_$2.csv" <<'PY'
import csv, sys
def load(p):
    return [r for r in csv.DictReader(open(open(p).read().strip())) if r["OP CODE"].startswith("GenericOp")]
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
exit 0
fi
stem=$(basename "$2" .py)
out=$(GENIFY_MATMUL_PERF=1 timeout 580 scripts/run_safe_pytest.sh --no-jit-server --run-all --profile "$2" 2>&1)
echo "$out" | grep -E "^(=+ ).*(passed|failed)"
echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //' > /tmp/t2p1_guard_$1_$stem.csv
cat /tmp/t2p1_guard_$1_$stem.csv
