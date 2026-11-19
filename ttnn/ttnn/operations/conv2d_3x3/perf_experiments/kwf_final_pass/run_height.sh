#!/bin/bash
# usage: KFP_SEQ=a,b run_height.sh <label>: profiles test_kfp_height_perf.py (14 HEIGHT shapes x KFP_SEQ) in one process.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_final_pass
label=$1; shift
timeout 580 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend $E/test_kfp_height_perf.py "$@" > $E/runs/log_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_$label.txt | grep -E " passed| failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError" | tail -6
csv=$(grep "PROFILER CSV:" $E/runs/log_$label.txt | sed 's/.*CSV: //')
python3 - "$csv" "$label" "$E" <<'PY'
import sys, csv, os
path, label, E = sys.argv[1:4]
if not path:
    print("no profiler csv"); sys.exit()
rows = [r for r in csv.DictReader(open(path)) if r.get("DEVICE KERNEL DURATION [ns]")]
v = [float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in rows]
shapes = "stem_conv1 stem_conv3 res2_conv2 res3_0_conv2 res3_conv2 sem_res2_fuse_conv0 stem_conv2 sem_res3_fuse_conv0 inst_res3_fuse_conv0 sem_res3_fuse_conv1 inst_res2_fuse_conv0 sem_res2_fuse_conv1 inst_res2_fuse_conv1 center_head1".split()
seq = os.environ.get("KFP_SEQ", "op_base_unz,op_copy3_unz").split(",")
ids = [(c, s) for c in shapes for s in seq]
if len(ids) != len(v):
    print("MISMATCH", len(ids), len(v), [round(x, 2) for x in v]); sys.exit()
with open(f"{E}/runs/results.txt", "a") as f:
    for (c, s), x in zip(ids, v):
        f.write(f"{label} {c} {s} {x:.2f}\n")
for c in shapes:
    print(f"{label:12s} {c:22s} " + "  ".join(f"{s}={x:.2f}" for (cc, s), x in zip(ids, v) if cc == c))
PY
