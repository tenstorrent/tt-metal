#!/bin/bash
# Tournament-2 Perf 1: alternate base (op files stashed) / new runs of single perf test ids, N reps.
# usage: t2p1_ab.sh <reps> "<file>::<id>" ...   -> label id us
cd /localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
source python_env/bin/activate
OPD=ttnn/ttnn/operations/genify_matmul
reps=$1; shift
one() {
  out=$(GENIFY_MATMUL_PERF=1 timeout 300 scripts/run_safe_pytest.sh --no-jit-server --profile "$2" 2>&1)
  csv=$(echo "$out" | grep "PROFILER CSV" | sed 's/.*CSV: //')
  us=$(python3 -c "import csv,sys; r=[x for x in csv.DictReader(open('$csv')) if x['OP CODE'].startswith('GenericOp')]; print(' '.join(f\"{int(x['DEVICE KERNEL DURATION [ns]'])/1000:.1f}\" for x in r))")
  echo "$1 $(basename $2) $us"
}
for r in $(seq $reps); do
  for t in "$@"; do one new "$t"; done
  git stash push -q -- $OPD/genify_matmul_program_descriptor.py $OPD/kernels
  for t in "$@"; do one base "$t"; done
  git stash pop -q
done
