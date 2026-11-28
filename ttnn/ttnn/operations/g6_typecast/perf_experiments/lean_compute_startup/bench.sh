#!/bin/bash
# usage: bench.sh <variant|op|stock> <case> [<case>...]   (variant = kernels_<variant>/ in this dir)
# One profiler session per case; median of launches 2..12 (same reader as _coord/bench.sh).
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip1; source python_env/bin/activate 2>/dev/null; export TT_METAL_HOME=$PWD
D=ttnn/ttnn/operations/g6_typecast/perf_experiments/lean_compute_startup
v=$1; shift
side=op; [ "$v" = stock ] && side=stock
for c in "$@"; do
  log=$(mktemp /tmp/lcsbench.XXXX)
  LCS_VARIANT=$v BENCH_SIDE=$side BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/bench_lcs.py::test_bench --import-mode=prepend > $log 2>&1
  csv=$(grep -o '/[^ ]*ops_perf_results_[^ ]*\.csv' $log | tail -1)
  ok=$(grep -c "BENCH_OK\|BENCH_STOCK_CHECK" $log)
  if [ -z "$csv" ]; then echo "$c $v FAIL (log $log)"; grep -E "Error|error:|FAILED|assert" $log | head -8; continue; fi
  line=$(python3 .claude/skills/tooling/scripts/ops_csv.py summary $csv --skip-first 1 2>/dev/null | grep -E "GenericOp|Typecast" | head -1)
  echo "$c $v ok=$ok $line" | awk '{printf "%-6s %-10s %s cores=%s med=%s B=%s N=%s T0=%s T1=%s T2=%s\n",$1,$2,$3,$6,$8,$14,$15,$16,$17,$18}'
  mkdir -p $D/logs; mv $log $D/logs/${c}_${v}.log
done
