#!/bin/bash
# Perf 2: usage prof.sh <tag> [env assignments...] — trace-replay profile of the real op on the reserved node.
TAG=$1; shift
P=ttnn/ttnn/operations/a2a_wkv_b2/perf/p2
cd /data/dstoiljkovic/tt-metal
/data/dstoiljkovic/bin/on-node "env $* TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/a2a_wkv_b2/probes/probe_trace_profile.py && cp generated/profiler/.logs/profile_log_device.csv $P/$TAG.csv" 2>&1 | grep -E "passed|failed|rror" | tail -3
source python_env/bin/activate
python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/trace_gaps.py $P/$TAG.csv | tail -6
python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/stage_zones.py $P/$TAG.csv
