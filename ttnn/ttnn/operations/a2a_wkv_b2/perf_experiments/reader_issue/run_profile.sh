#!/bin/bash
# usage: run_profile.sh <tag> [env assignments...]; runs the trace profile probe on the reserved node, copies the csv
TAG=$1; shift
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/reader_issue
mkdir -p $D/profiles
cd /data/dstoiljkovic/tt-metal
/data/dstoiljkovic/bin/on-node "env $* TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh $D/probe_ri_trace_profile.py --import-mode=prepend && cp generated/profiler/.logs/profile_log_device.csv $D/profiles/$TAG.csv" 2>&1 | tail -4
date -u +%FT%TZ
source python_env/bin/activate; python3 $D/ri_reader_ends.py $D/profiles/$TAG.csv --all
