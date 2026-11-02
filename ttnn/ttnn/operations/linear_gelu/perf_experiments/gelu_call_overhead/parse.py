"""Pair linear_gelu rows of the newest ops_perf_results CSV with gen/calls.txt (call order). Usage: python parse.py"""
import csv
import glob
import os
import sys
from pathlib import Path

calls = [l.split() for l in (Path(__file__).parent / "gen" / "calls.txt").read_text().splitlines() if l.strip()]
path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else max(
        glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"),
        key=os.path.getmtime,
    )
)
rows = list(csv.reader(open(path)))
hdr = rows[0]
col = hdr.index("DEVICE KERNEL DURATION [ns]")
code = hdr.index("OP CODE")
ops = [r for r in rows[1:] if r[col].strip() and "Generic" in r[code] or "linear_gelu" in r[code].lower()]
print(path, f"{len(ops)} op rows, {len(calls)} calls")
for (name, shape), r in zip(calls, ops):
    print(f"{name:28s} {shape:8s} {int(float(r[col])):>9d} ns  {r[code]}")
