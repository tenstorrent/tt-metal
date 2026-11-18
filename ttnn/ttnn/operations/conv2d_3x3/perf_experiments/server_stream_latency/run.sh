#!/bin/bash
# usage: run.sh <label> <ids...>   ids = focus ids (res3_1_2_3_conv2 ...) run against the copy; env passes through
# (SSL_* variant switches). With SSL_REAL=1 the real op (tests/.../test_conv2d_3x3_perf.py) is profiled instead.
# Profiles in one process, prints DEVICE KERNEL DURATION per op (us), keeps the device CSV in runs/.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/server_stream_latency
label=$1; shift
nodes=()
for id in "$@"; do
    if [ "${SSL_REAL:-0}" = 1 ]; then
        nodes+=("tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus[$id]")
    else
        nodes+=("$E/test_ssl_op.py::test_conv2d_3x3_perf_focus[$id]")
    fi
done
out=$(timeout 1800 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "${nodes[@]}" 2>&1)
echo "$out" | grep -E " passed| failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError" | tail -8
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp generated/profiler/.logs/profile_log_device.csv $E/runs/dev_$label.csv 2>/dev/null
echo "$out" > $E/runs/log_$label.txt
python3 - "$csv" "$label" "$@" <<'PY'
import sys, csv, json
path, label, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
if not path:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(path)))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 2) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
print("DKD_us", label, v)
with open("ttnn/ttnn/operations/conv2d_3x3/perf_experiments/server_stream_latency/results.jsonl", "a") as f:
    f.write(json.dumps({"label": label, "ids": ids, "us": v}) + "\n")
PY
