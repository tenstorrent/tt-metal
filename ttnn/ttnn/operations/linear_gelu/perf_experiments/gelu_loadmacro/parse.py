"""Print DEVICE KERNEL DURATION of the GenericOp / linear_gelu rows of the newest ops_perf_results CSV, in order,
paired with LAUNCH/E2E labels read from stdin (pipe run.sh output in). Usage: run.sh ... | python parse.py"""
import csv
import glob
import os
import sys

labels = [l.strip() for l in sys.stdin if l.startswith(("LAUNCH", "E2E"))]
path = max(
    glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime
)
rows = list(csv.reader(open(path)))
hdr = rows[0]
col = hdr.index("DEVICE KERNEL DURATION [ns]")
code = hdr.index("OP CODE")
ops = [r for r in rows[1:] if r[col].strip() and ("Generic" in r[code] or "linear_gelu" in r[code].lower())]
print(path, f"{len(ops)} op rows, {len(labels)} labels")
labels = labels[len(labels) - len(ops) :]  # drop the precompile collect pass labels
for lab, r in zip(labels, ops):
    print(f"{lab:60s} {int(float(r[col])):>10d} ns")
