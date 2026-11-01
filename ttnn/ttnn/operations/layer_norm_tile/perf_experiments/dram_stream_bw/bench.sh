#!/bin/bash
# bench.sh <label> [LN_KNOBS] [-k expr] [LN_COPY]: one profiled pass over perf+guard rows of the copy; summary -> logs/<label>.txt
cd /localdev/kgrujcic/tt-metal-codegen
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/dram_stream_bw
mkdir -p $here/logs
k=${3:-perf_case or guard_case}
LN_ABLATE="$LN_ABLATE" PYTEST_ADDOPTS="--import-mode=prepend -k '$k'" LN_COPY="${4:-1}" LN_KNOBS="$2" scripts/run_safe_pytest.sh --run-all --profile $here/test_dram_stream_bw.py > $here/logs/last_run.log 2>&1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
{ echo "== $1 [$2] k=$k copy=${4:-1}"; python3 $here/ops.py "$csv" $here/logs/last_run.log; grep -E "passed|failed" $here/logs/last_run.log | tail -1; } | tee -a $here/logs/$1.txt
