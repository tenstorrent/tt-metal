#!/bin/bash
# bench.sh <label> [LN_KNOBS] [-k filter]: one profiled pass over the perf + guard rows, summary appended to logs/<label>.txt
cd /localdev/kgrujcic/tt-metal-codegen
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/half_band
mkdir -p /tmp/ln4 $here/logs
PYTEST_ADDOPTS="${3:+-k '$3'}" LN_KNOBS="$2" scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py > /tmp/ln4/last_run.log 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
python3 ttnn/ttnn/operations/layer_norm_tile/perf_experiments/stream_stats/ops.py $csv | tee -a $here/logs/$1.txt
grep -E "passed|failed" /tmp/ln4/last_run.log | tail -1
