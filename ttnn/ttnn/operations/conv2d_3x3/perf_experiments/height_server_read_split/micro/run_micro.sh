#!/bin/bash
# usage: run_micro.sh <label> [pytest -k expr]: profile the micro-bench, print per-config DKD and per-engine mb_read.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
M=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/height_server_read_split/micro
label=$1; shift
out=$(timeout 1800 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend $M/test_micro.py "$@" 2>&1)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|Error" | tail -5
echo "$out" > $M/log_$label.txt
cp generated/profiler/.logs/profile_log_device.csv $M/dev_$label.csv && gzip -f $M/dev_$label.csv
echo "$out" | grep -o "test_micro\[[^]]*\]" | uniq > $M/ids_$label.txt
python3 $M/report_micro.py $M/dev_$label.csv.gz $M/ids_$label.txt | tee $M/report_$label.txt
