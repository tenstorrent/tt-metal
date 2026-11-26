#!/bin/bash
# usage: prof.sh <tag> [env assignments, e.g. TXE_VARIANT=base]; trace-replay device profile of this dir's variant
# on the reserved node -> profiles/<tag>.csv, then the op's stage_zones.py + perf/p2/tail.py.
TAG=$1; shift
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/tx_early_start
P=ttnn/ttnn/operations/a2a_wkv_b2/perf
cd /data/dstoiljkovic/tt-metal
/data/dstoiljkovic/bin/on-node "env $* TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh $D/probe_txe_trace_profile.py -o addopts= --import-mode=prepend && cp generated/profiler/.logs/profile_log_device.csv $D/profiles/$TAG.csv" 2>&1 | grep -E "passed|failed|rror" | tail -3
source python_env/bin/activate
python3 $P/trace_gaps.py $D/profiles/$TAG.csv | tail -4
python3 $P/stage_zones.py $D/profiles/$TAG.csv | grep -E "span|tx_|marker"
python3 $P/p2/tail.py $D/profiles/$TAG.csv | sed -n '/tx cores/,$p'
