#!/bin/bash
# usage: zones.sh <label> [env...]
cd /home/malimpic/localdev/2026_10_30/1730_malimpic_nuke-ccl-ops/clones/distributed_rms_norm_run1/tt-metal
source python_env/bin/activate
label=$1; shift
d=/tmp/esw/prof/$label; rm -rf $d; mkdir -p $d
env "$@" TT_METAL_PROFILER_DIR=$d DRMS_ZONES=1 timeout 3000 scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/early_small_wave/perf_zones_esw.py -s > $d/log.txt 2>&1
grep SAFE_PYTEST_RESULT $d/log.txt
python ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/early_small_wave/tools/esw_tl.py $d/.logs/profile_log_device.csv 0,1 > $d/tl.txt 2>&1
