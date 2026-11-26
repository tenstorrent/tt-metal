#!/bin/bash
# usage: run_profile.sh <tag> [env assignments...]; trace-replay device profile of this dir's variant on the
# reserved node, csv copied to profiles/<tag>.csv, then summarised (op's perf/stage_zones.py + txs_zones.py).
TAG=$1; shift
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/tx_startup
cd /data/dstoiljkovic/tt-metal
/data/dstoiljkovic/bin/on-node "env $* TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh $D/probe_txs_trace_profile.py && cp generated/profiler/.logs/profile_log_device.csv $D/profiles/$TAG.csv" 2>&1 | grep -E "SAFE_PYTEST_RESULT|passed|failed" | tail -3
source python_env/bin/activate
python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/stage_zones.py $D/profiles/$TAG.csv | grep -E "span|tx_|marker"
python3 $D/txs_zones.py $D/profiles/$TAG.csv
