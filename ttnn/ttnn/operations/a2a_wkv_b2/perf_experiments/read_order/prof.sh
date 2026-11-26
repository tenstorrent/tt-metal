#!/bin/bash
# read_order: usage prof.sh <tag> [env assignments...] — trace-replay profile of the read_order variant.
TAG=$1; shift
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/read_order
cd /data/dstoiljkovic/tt-metal
/data/dstoiljkovic/bin/on-node "env $* TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh $D/probe_ro_trace_profile.py -o addopts= --import-mode=prepend && cp generated/profiler/.logs/profile_log_device.csv $D/profiles/$TAG.csv" 2>&1 | grep -E "passed|failed|rror" | tail -3
source python_env/bin/activate
python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/trace_gaps.py $D/profiles/$TAG.csv | tail -6
python3 $D/ro_tail.py $D/profiles/$TAG.csv
