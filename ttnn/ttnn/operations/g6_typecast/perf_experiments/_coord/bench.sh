#!/bin/bash
# usage: bench.sh <side op|stock> <case> [<case>...]  -> prints "case side median_ns cores per-RISC..." per case
# One profiler session per case (cases can share a program hash). Median of launches 2..12.
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip1; source python_env/bin/activate; export TT_METAL_HOME=$PWD
side=$1; shift
for c in "$@"; do
  log=$(mktemp /tmp/g6bench.XXXX)
  BENCH_SIDE=$side BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones generated/meta/g6_typecast/bench/bench_cases.py::test_bench > $log 2>&1
  csv=$(grep -o '/[^ ]*ops_perf_results_[^ ]*\.csv' $log | tail -1)
  ok=$(grep -c "BENCH_OK\|BENCH_STOCK_CHECK" $log)
  if [ -z "$csv" ]; then echo "$c $side FAIL (log $log)"; tail -5 $log; continue; fi
  line=$(python3 .claude/skills/tooling/scripts/ops_csv.py summary $csv --skip-first 1 2>/dev/null | grep -E "GenericOp|Typecast" | head -1)
  echo "$c $side ok=$ok $line" | awk '{printf "%-6s %-5s %s cores=%s med=%s B=%s N=%s T0=%s T1=%s T2=%s\n",$1,$2,$3,$6,$8,$14,$15,$16,$17,$18}'
  rm -f $log
done
