#!/bin/bash
# usage: focus.sh <label>: the 22 focus cases on the working tree, zoned, PCC asserted; appends to /tmp/abl15/focus.txt
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
label=$1
out=$(timeout 1800 scripts/run_safe_pytest.sh --profile --run-all "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus" 2>&1)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|^FAILED|HANG|triage" | tail -5
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp generated/profiler/.logs/profile_log_device.csv /tmp/abl15/focus_$label.csv
python3 - "$csv" "$label" <<'PY'
import sys, pandas as pd
df=pd.read_csv(sys.argv[1]); d=df[df["DEVICE KERNEL DURATION [ns]"].notna()]
v=[round(x/1000,2) for x in d["DEVICE KERNEL DURATION [ns]"].tolist()]
print("FOCUS", sys.argv[2], v)
open("/tmp/abl15/focus.txt","a").write(f"{sys.argv[2]} {v}\n")
PY
