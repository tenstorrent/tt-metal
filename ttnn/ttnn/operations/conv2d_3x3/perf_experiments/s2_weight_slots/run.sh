#!/bin/bash
# usage: [SWS_CASES=.. SWS_VARIANTS=.. SWS_REPS=.. SWS_ORDER=alt SWS_UNZ=1] run.sh <label> [--noprof]
# One profiled session of test_sws_ab.py; prints DEVICE KERNEL DURATION per call (joined with the call log), keeps the
# ops CSV and the gzipped device log in runs/, appends to results.txt.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/s2_weight_slots
mkdir -p $E/runs
label=$1
export SWS_CALLS=calls_$label.txt
rm -f $E/runs/calls_$label.txt
python3 $E/zone_hash_check.py || exit 1
PROF="--profile"; [ "$2" == "--noprof" ] && PROF=""
timeout 2700 scripts/run_safe_pytest.sh $PROF --run-all --no-precompile --import-mode=prepend -s $E/test_sws_ab.py > $E/runs/log_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_$label.txt | grep -E "passed|failed|error:|Error|SAFE_PYTEST_RESULT|HANG|triage|AssertionError|colliding|SWS_INFO|SWS_CBS" | tail -40
[ -z "$PROF" ] && exit 0
csv=$(grep "PROFILER CSV:" $E/runs/log_$label.txt | sed 's/.*CSV: //')
[ -f generated/profiler/.logs/profile_log_device.csv ] && gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/dev_$label.csv.gz
[ -n "$csv" ] && cp "$csv" $E/runs/ops_$label.csv
python3 - "$csv" "$label" "$E/runs/calls_$label.txt" "$E/results.txt" <<'PY'
import sys, csv
if not sys.argv[1]:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(sys.argv[1])))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 3) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
calls = [l.split() for l in open(sys.argv[3])]
with open(sys.argv[4], "a") as f:
    for c, d in zip(calls, v):
        line = f"{sys.argv[2]} {' '.join(c)} {d}"
        print(line); f.write(line + "\n")
if len(calls) != len(v):
    print("WARNING calls", len(calls), "rows", len(v))
PY
