#!/bin/bash
# usage: ab_focus.sh <label>: the 22 focus cases on the WORKING TREE with zones stripped (restored after); appends
# unzoned DEVICE KERNEL DURATION (us) to /tmp/abl15/ab_focus.txt.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
OD=ttnn/ttnn/operations/conv2d_3x3
label=$1
bk=$(mktemp -d)
cp -r $OD/kernels $bk/
sed -i '/MaybeDeviceZoneScope(/d' $OD/kernels/*.cpp $OD/kernels/*.hpp
out=$(timeout 1800 scripts/run_safe_pytest.sh --profile --run-all "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus" 2>&1)
rm -rf $OD/kernels && cp -r $bk/kernels $OD/ && rm -rf $bk
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|^FAILED" | tail -3
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
python3 - "$csv" "$label" <<'PY'
import sys, pandas as pd
df=pd.read_csv(sys.argv[1]); d=df[df["DEVICE KERNEL DURATION [ns]"].notna()]
v=[round(x/1000,3) for x in d["DEVICE KERNEL DURATION [ns]"].tolist()]
print("ABF", sys.argv[2], v)
open("/tmp/abl15/ab_focus.txt","a").write(f"{sys.argv[2]} {v}\n")
PY
