#!/usr/bin/env bash
# bench.sh <case> [impl]  -> median DEVICE KERNEL DURATION of launches 2..N, plus zone medians (op only)
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip0
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; unset TT_METAL_SIMULATOR TT_METAL_PROFILER_DIR
source python_env/bin/activate
c=$1; impl=${2:-new}
L=ttnn/ttnn/operations/g6_reshape/perf_experiments/_coord/logs; mkdir -p $L
for try in 1 2; do
  BENCH_CASES=$c BENCH_IMPL=$impl TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones generated/meta/g6_reshape/bench/bench_cases.py::test_bench > $L/${c}_${impl}.log 2>&1
  csv=$(grep -o 'PROFILER CSV: .*' $L/${c}_${impl}.log | sed 's/PROFILER CSV: //')
  [ -n "$csv" ] && break
done
grep -E 'BENCH_OK|FAIL|Error' $L/${c}_${impl}.log | head -3
python3 ttnn/ttnn/operations/g6_reshape/perf_experiments/_coord/parse.py $L/${c}_${impl}.log $c $impl
echo CSV $csv
