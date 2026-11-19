#!/bin/bash
# usage: run.sh <label> <node...>   node = a focus id (res3_{1,2,3}_conv2 ...) or H:<height-perf id> (res3_conv2 ...),
# run against this experiment's op copy; env passes through (HRL_* variant switches). With HRL_REAL=1 the real op
# (tests/.../test_conv2d_3x3_perf.py at HEAD) is profiled instead. Profiles in one process, prints DEVICE KERNEL
# DURATION per op (us), keeps the device CSV in runs/ (gzipped) and appends to results.jsonl.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/height_round0_latency
label=$1; shift
nodes=()
for id in "$@"; do
    if [[ $id == H:* ]]; then t=test_conv2d_3x3_height_sharded_perf; id=${id#H:}; elif [[ $id == I:* ]]; then t=test_conv2d_3x3_perf; id=${id#I:}; else t=test_conv2d_3x3_perf_focus; fi
    if [ "${HRL_REAL:-0}" = 1 ]; then
        nodes+=("tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::$t[$id]")
    else
        nodes+=("$E/test_hrl_op.py::$t[$id]")
    fi
done
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "${nodes[@]}" 2>&1)
echo "$out" | grep -E " passed| failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError|PCC" | tail -8
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp generated/profiler/.logs/profile_log_device.csv $E/runs/dev_$label.csv 2>/dev/null && gzip -f $E/runs/dev_$label.csv
echo "$out" > $E/runs/log_$label.txt
env | grep "^HRL_" | sort | tr '\n' ' ' > $E/runs/env_$label.txt
python3 - "$csv" "$label" "$E" "$@" <<'PY'
import sys, csv, json, os
path, label, E, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
if not path:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(path)))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 3) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
env = {k: v for k, v in os.environ.items() if k.startswith("HRL_")}
print("DKD_us", label, env, v)
with open(f"{E}/results.jsonl", "a") as f:
    f.write(json.dumps({"label": label, "env": env, "ids": ids, "us": v}) + "\n")
PY
