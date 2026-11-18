#!/bin/bash
# usage: run.sh <label> <pytest node ids...>   (env passes through: KWCR_FORCE_KWF, KWCR_COPY, ...)
# Profiles the given node ids in one process, prints DEVICE KERNEL DURATION per op, and keeps both CSVs in runs/.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds
label=$1; shift
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "$@" 2>&1)
echo "$out" | grep -E "passed|failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError|PCC" | tail -12
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
cp generated/profiler/.logs/profile_log_device.csv $E/runs/dev_$label.csv 2>/dev/null
[ -n "$csv" ] && cp "$csv" $E/runs/ops_$label.csv
echo "$out" > $E/runs/log_$label.txt
python3 - "$csv" "$label" <<'PY'
import sys, csv
if not sys.argv[1]:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(sys.argv[1])))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 2) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
print("DKD_us", sys.argv[2], v)
open("ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds/runs/results.txt", "a").write(
    f"{sys.argv[2]} {v}\n")
PY
