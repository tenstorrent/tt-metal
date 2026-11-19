#!/bin/bash
# usage: KFP_SEQ=a,b [KFP_CASES=..] run.sh <label>: profiles test_kfp_perf.py in one process, prints DEVICE KERNEL
# DURATION per (case, variant) and appends them to runs/results.txt; keeps the device CSV in runs/.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_final_pass
mkdir -p $E/runs
label=$1; shift
timeout 2400 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend $E/test_kfp_perf.py "$@" > $E/runs/log_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_$label.txt | grep -E " passed| failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError" | tail -6
csv=$(grep "PROFILER CSV:" $E/runs/log_$label.txt | sed 's/.*CSV: //')
cp generated/profiler/.logs/profile_log_device.csv $E/runs/dev_$label.csv 2>/dev/null && gzip -f $E/runs/dev_$label.csv
python3 - "$csv" "$label" "$E" <<'PY'
import sys, csv, os
path, label, E = sys.argv[1:4]
if not path:
    print("no profiler csv"); sys.exit()
rows = [r for r in csv.DictReader(open(path)) if r.get("DEVICE KERNEL DURATION [ns]")]
v = [float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in rows]
cases = os.environ.get("KFP_CASES", "stem_conv2,stem_conv3,res2,res3x,sem_dec_res3_fc0,inst_dec_res3_fc0,sem_dec_res3_fc1,sem_dec_res2_fc0,inst_dec_res2_fc0,sem_res2_fc1,inst_res2_fc1,center_head1").split(",")
seq = os.environ.get("KFP_SEQ", "op_base_unz,op_copy_unz").split(",")
ids = [(c, s) for c in cases for s in seq]
if len(ids) != len(v):
    print("MISMATCH", len(ids), len(v), [round(x, 2) for x in v]); sys.exit()
with open(f"{E}/runs/results.txt", "a") as f:
    for (c, s), x in zip(ids, v):
        f.write(f"{label} {c} {s} {x:.2f}\n")
for c in cases:
    print(f"{label:12s} {c:20s} " + "  ".join(f"{s}={x:.2f}" for (cc, s), x in zip(ids, v) if cc == c))
PY
