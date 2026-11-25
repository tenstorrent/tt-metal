#!/bin/bash
# s1_pace domain sweep: LOOSE_CASES 1..3, one ccl_perf run (5 exp) each.  sweep.sh <pkg> [knobs]
cd "$(dirname "$0")/../../../../../.."
source python_env/bin/activate
S=ttnn/ttnn/operations/q_stem/perf_experiments/s1_pace
OUT=$S/results/sweep; mkdir -p $OUT
QSTEM_CASES=1,2,3 QSTEM_MODE=perf QSTEM_PKG=$1 QSTEM_KNOBS="$2" CCL_PERF_OUT=$OUT timeout 590 \
  scripts/run_safe_pytest.sh --no-precompile --run-all --import-mode=prepend $S/bench.py::test_bench -s > $OUT/last_$1.log 2>&1
grep -E "SAFE_PYTEST_RESULT|passed|failed" $OUT/last_$1.log | tail -1
tag=$(echo "$2" | sed 's/;/_/g; s/=//g; s/ //g; s/,/-/g; s/(//g; s/)//g')
for c in 1 2 3; do python -m eval.ccl_perf show $OUT/$1${tag:+__$tag}__lc$c.json | sed -n '1,3p'; done
