#!/bin/bash
# usage: NB_CELLS=a,b NB_VARIANTS=x,y run.sh <tag>   (from anywhere; foreground only)
# Runs the bench under the profiler, copies the ops CSV + device log into runs/<tag>.*, prints the table.
set -u
ROOT=/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
MY=$ROOT/ttnn/ttnn/operations/rms_norm/perf_experiments/dram_noc_balance
cd $ROOT && source python_env/bin/activate
mkdir -p $MY/runs
out=$(scripts/run_safe_pytest.sh --profile $MY/bench_noc_balance.py -s --import-mode=prepend 2>&1); csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p')
cp "$csv" $MY/runs/$1.ops.csv; cp generated/profiler/.logs/profile_log_device.csv $MY/runs/$1.dev.csv
echo "$out" | grep -E "^NB |SAFE_PYTEST_RESULT|Error|FAILED|assert" | head -60
echo "$out" > $MY/runs/$1.log
python3 - "$MY/runs/$1.ops.csv" <<'PY'
import csv, os, sys
cells = os.environ["NB_CELLS"].split(","); vs = ["base"] + os.environ["NB_VARIANTS"].split(",")
g = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
i = 0
for c in cells:
    b = None
    for v in vs:
        if i >= len(g): break
        ns = int(g[i]["DEVICE KERNEL DURATION [ns]"]); i += 1
        b = b or ns
        print(f"{c:24s} {v:12s} {ns:>9d} ns  {b/ns:5.3f}x")
PY
