#!/bin/bash
# usage: run_peel.sh <label> <tokens>   (tokens: comma list of gather,fill,weights,scatter,untilize,tilize,matmul; "" = full)
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
mkdir -p /tmp/abl11
label=$1; toks=$2
out=$(CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=$toks timeout 1800 scripts/run_safe_pytest.sh --profile --run-all "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus" 2>&1)
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
echo "$out" | grep -E "passed|failed|SAFE_PYTEST_RESULT|HANG|triage" | tail -3
cp generated/profiler/.logs/profile_log_device.csv /tmp/abl11/prof_$label.csv
python3 - "$csv" "$label" <<'PY'
import sys, pandas as pd
df=pd.read_csv(sys.argv[1]); d=df[df["DEVICE KERNEL DURATION [ns]"].notna()]
v=[round(x/1000,1) for x in d["DEVICE KERNEL DURATION [ns]"].tolist()]
print("PEEL", sys.argv[2], v)
open("/tmp/abl11/peel_results.txt","a").write(f"{sys.argv[2]} {v}\n")
PY
