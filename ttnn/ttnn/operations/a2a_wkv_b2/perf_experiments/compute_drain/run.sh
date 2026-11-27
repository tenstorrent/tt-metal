#!/bin/bash
# usage: run.sh <tag> <variants comma list> [gap_ns]  — profile the bench on the reserved node, summarise.
TAG=$1; VARS=$2; GAP=${3:-1900}
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/compute_drain
cd /data/dstoiljkovic/tt-metal
mkdir -p $D/profiles
/data/dstoiljkovic/bin/on-node "env CD_VARIANTS=$VARS CD_ROW_GAP_NS=$GAP TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --run-all $D/test_cd_bench.py -o addopts= --import-mode=prepend -s && cp generated/profiler/.logs/profile_log_device.csv $D/profiles/$TAG.csv" 2>&1 | grep -E "CD_RESULT|passed|failed|rror|SAFE_PYTEST" | tail -20
source python_env/bin/activate
python3 $D/summarize.py $D/profiles/$TAG.csv $VARS
