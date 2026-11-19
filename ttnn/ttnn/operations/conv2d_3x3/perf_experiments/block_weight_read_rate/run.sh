#!/bin/bash
# usage: run.sh <label> <node...>   node = focus:<id> | block:<layer>   (env BWR_* picks the variant, passed through)
# Profiles in one process against the op copy, prints DEVICE KERNEL DURATION per op (us), keeps the device log in runs/.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_weight_read_rate
label=$1; shift
nodes=()
for n in "$@"; do
    case $n in
        focus:*) nodes+=("$E/test_bwr_perf.py::test_conv2d_3x3_perf_focus[${n#focus:}]") ;;
        block:*) nodes+=("$E/test_bwr_perf.py::test_conv2d_3x3_block_sharded_perf[${n#block:}]") ;;
        *) echo "bad node $n"; exit 1 ;;
    esac
done
export BWR_LOG=${BWR_LOG:-precision.jsonl}
out=$(timeout 1800 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "${nodes[@]}" 2>&1)
echo "$out" | grep -E " passed| failed|error:|Error|SAFE_PYTEST_RESULT|HANG|triage|AssertionError" | tail -8
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/dev_$label.csv.gz 2>/dev/null
echo "$out" > $E/runs/log_$label.txt
python3 - "$csv" "$label" "$@" <<'PY'
import sys, csv, json, os
path, label, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
if not path:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(path)))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 2) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
print("DKD_us", label, v)
env = {k: v for k, v in os.environ.items() if k.startswith("BWR_") or k.startswith("CONV2D_3X3_ABLATE")}
with open("ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_weight_read_rate/results.jsonl", "a") as f:
    f.write(json.dumps({"label": label, "ids": ids, "us": v, "env": env}) + "\n")
PY
