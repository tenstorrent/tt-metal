#!/bin/bash
# usage: run.sh <label> <pytest node ids...>   (env passes through: XSS_OP=base|copy|base_unz|copy_unz)
# Profiles the given node ids in one process, prints DEVICE KERNEL DURATION per op, keeps the CSVs in runs/.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_xchg_slab_skip
mkdir -p $E/runs
label=$1; shift
timeout 2400 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "$@" > $E/runs/log_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_$label.txt | grep -E "passed|failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError|colliding" | tail -8
csv=$(grep "PROFILER CSV:" $E/runs/log_$label.txt | sed 's/.*CSV: //')
cp generated/profiler/.logs/profile_log_device.csv $E/runs/dev_$label.csv 2>/dev/null && gzip -f $E/runs/dev_$label.csv
[ -n "$csv" ] && cp "$csv" $E/runs/ops_$label.csv
python3 - "$csv" "$label" <<'PY'
import sys, csv
if not sys.argv[1]:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(sys.argv[1])))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 2) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
print("DKD_us", sys.argv[2], v)
open("/localdev/pmilojevic/workspaces/opgen/ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_xchg_slab_skip/runs/dkd.txt", "a").write(f"{sys.argv[2]} {v}\n")
PY
