#!/bin/bash
# usage: sweep.sh <suite> <tag> [POLICY]   one run of every case of the suite: real kernels (POLICY empty) or the PV policy
# writes sweep_<suite>_<tag>.txt : "<case id> <ns> <placement>"
D=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_tail_write_stream
cd $D
if [ -z "$3" ]; then TWS_REAL=1 ./tws_run.sh $1 "" "" "" > /dev/null 2>&1; else POLICY=$3 ./tws_run.sh $1 "" "" "" > /dev/null 2>&1; fi
python3 - "$1" "$2" <<'PY'
import csv, glob, os, re, sys
D = os.getcwd()
log = open(f"{D}/last_run.log").read().splitlines()
# real-pass PLAN lines: the last occurrence block (precompile collect pass prints them first)
plans = [l for l in log if l.startswith("PLAN ")]
ids, place = [], {}
for l in plans:
    cid = l.split()[1]
    m = re.search(r"placement='(\w+)'", l)
    place[cid] = m.group(1) if m else "?"
n_real = None
csvs = sorted(glob.glob(f"{D}/prof/reports/*/ops_perf_results*.csv"))
rows = [r for r in csv.DictReader(open(csvs[-1])) if "generic" in r.get("OP CODE", "").lower()]
real_ids = [l.split()[1] for l in plans][-len(rows):]
with open(f"{D}/sweep_{sys.argv[1]}_{sys.argv[2]}.txt", "w") as f:
    for cid, r in zip(real_ids, rows):
        line = f"{cid:28s} {int(r['DEVICE KERNEL DURATION [ns]']):9d} {place.get(cid)}"
        print(line); f.write(line + "\n")
PY
grep -E "^=+ .*(passed|failed)" $D/last_run.log | tail -1
