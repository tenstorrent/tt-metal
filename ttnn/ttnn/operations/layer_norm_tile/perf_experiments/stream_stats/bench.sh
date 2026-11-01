#!/bin/bash
# bench.sh <label> [LN_KNOBS] : one profiled pass over the perf + guard rows, summary appended to logs/<label>.txt
cd /localdev/kgrujcic/tt-metal-codegen
mkdir -p ttnn/ttnn/operations/layer_norm_tile/perf_experiments/stream_stats/logs
LN_KNOBS="$2" scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py > /tmp/ln3/last_run.log 2>&1
python3 ttnn/ttnn/operations/layer_norm_tile/perf_experiments/stream_stats/ops.py | tee -a ttnn/ttnn/operations/layer_norm_tile/perf_experiments/stream_stats/logs/$1.txt
grep -E "passed|failed" /tmp/ln3/last_run.log | tail -1
cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv /tmp/ln3/$1.device.csv
