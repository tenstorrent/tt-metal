#!/bin/bash
# usage: ab_unzoned.sh <rev> <label>: check out the op at <rev>, strip zones, profile the whole perf file, restore HEAD.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
OD=ttnn/ttnn/operations/conv2d_3x3
rev=$1; label=$2
git checkout $rev -- $OD/conv2d_3x3.py $OD/conv2d_3x3_program_descriptor.py $OD/kernels
sed -i '/MaybeDeviceZoneScope(/d' $OD/kernels/*.cpp $OD/kernels/*.hpp
out=$(timeout 3000 scripts/run_safe_pytest.sh --profile --run-all "${@:3}" 2>&1)
git checkout HEAD -- $OD/conv2d_3x3.py $OD/conv2d_3x3_program_descriptor.py $OD/kernels
git status --short $OD/kernels | head -3
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|^FAILED" | tail -3
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
python3 - "$csv" "$label" <<'PY'
import sys, pandas as pd
df=pd.read_csv(sys.argv[1]); d=df[df["DEVICE KERNEL DURATION [ns]"].notna()]
v=[round(x/1000,2) for x in d["DEVICE KERNEL DURATION [ns]"].tolist()]
print("AB", sys.argv[2], len(v))
open("/tmp/abl11/ab_guard.txt","a").write(f"{sys.argv[2]} {v}\n")
PY
