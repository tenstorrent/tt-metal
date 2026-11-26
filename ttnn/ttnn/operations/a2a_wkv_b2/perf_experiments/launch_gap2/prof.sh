#!/bin/bash
# usage (from the login node): prof.sh <variant> — trace-replay profile of one variant, prints per-chip gaps.
V=$1
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/launch_gap2
cd /data/dstoiljkovic/tt-metal
mkdir -p $D/prof
/data/dstoiljkovic/bin/on-node "env V_VARIANTS=$V TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --no-precompile $D/test_v_profile.py -o addopts= --import-mode=prepend && cp generated/profiler/.logs/profile_log_device.csv $D/prof/$V.csv" 2>&1 | grep -E "passed|failed|rror" | tail -3
source python_env/bin/activate
python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/trace_gaps.py $D/prof/$V.csv | tail -8
