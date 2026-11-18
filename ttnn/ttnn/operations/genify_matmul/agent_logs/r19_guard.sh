#!/bin/bash
# Refinement 19 guard: run every perf file (all cases, --profile) under a label; `cmp` prints base vs new per row.
# usage: r19_guard.sh <label> [files...] ; r19_guard.sh cmp <base label> <new label>
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
mkdir -p /tmp/r19g
P=tests/ttnn/unit_tests/operations/genify_matmul/perf
if [ "$1" == "cmp" ]; then
for f in /tmp/r19g/$2_*.csvpath; do
  stem=${f#/tmp/r19g/$2_}; stem=${stem%.csvpath}
  [ -f /tmp/r19g/$3_$stem.csvpath ] || continue
  echo "== $stem"
  python3 - "$f" "/tmp/r19g/$3_$stem.csvpath" <<'PY'
import csv, sys
def load(p):
    return [r for r in csv.DictReader(open(open(p).read().strip())) if r["OP CODE"].startswith("GenericOp")]
b, n = load(sys.argv[1]), load(sys.argv[2])
print(f"rows base {len(b)} new {len(n)}")
for i, (x, y) in enumerate(zip(b, n)):
    xb, yb = int(x["DEVICE KERNEL DURATION [ns]"]) / 1000, int(y["DEVICE KERNEL DURATION [ns]"]) / 1000
    d = (yb / xb - 1) * 100
    if abs(d) > 3:
        print(f"{i:3d} cores {x['CORE COUNT']:>4s}->{y['CORE COUNT']:>4s} {xb:9.1f} {yb:9.1f} {d:+6.1f}%" + (" <<" if d > 0 else " **"))
PY
done
exit 0
fi
label=$1; shift
files=${@:-$(ls $P/test_genify_matmul_perf_*.py)}
for f in $files; do
  stem=$(basename "$f" .py)
  out=$(GENIFY_MATMUL_PERF=1 timeout 1200 scripts/run_safe_pytest.sh --run-all --profile "$f" 2>&1)
  echo "$stem: $(echo "$out" | grep -E "^(=+ ).*(passed|failed)")"
  echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //' > /tmp/r19g/${label}_$stem.csvpath
done
