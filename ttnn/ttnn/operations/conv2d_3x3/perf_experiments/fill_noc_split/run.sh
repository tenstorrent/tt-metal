#!/bin/bash
# usage: run.sh <label> <test id> [<test id> ...]   e.g. run.sh smoke p1-base p1-split
# (explicit node ids: python -m tracy loses -k quoting.) Profiles the fill bench and prints one line per run
# (report.py). Logs / device CSV / ids under runs/<label>*.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
D=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fill_noc_split
mkdir -p $D/runs
label=$1; shift
nodes=()
for id in "$@"; do nodes+=("$D/test_fill_bench.py::test_fill[$id]"); done
out=$(timeout 3000 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "${nodes[@]}" 2>&1)
echo "$out" > $D/runs/log_$label.txt
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|Error|assert" | tail -8
cp generated/profiler/.logs/profile_log_device.csv $D/runs/dev_$label.csv && gzip -f $D/runs/dev_$label.csv
# ids in command-line order, minus the ones that failed before dispatch (FAILED summary lines)
failed=$(echo "$out" | grep -oE "^FAILED [^ ]*test_fill\[[^]]*\]" | grep -oE "\[[^]]*\]" | tr -d '[]')
for id in "$@"; do echo "$failed" | grep -qx -- "$id" || echo "$id"; done > $D/runs/ids_$label.txt
python3 $D/report.py $D/runs/dev_$label.csv.gz $D/runs/ids_$label.txt | tee $D/runs/report_$label.txt
