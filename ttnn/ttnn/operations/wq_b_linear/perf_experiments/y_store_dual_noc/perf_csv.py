"""Print DEVICE KERNEL DURATION [ns] per op row of an ops_perf_results CSV (default: newest)."""
import csv
import glob
import os
import sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
)
with open(path) as f:
    rows = list(csv.DictReader(f))
print(path)
for r in rows:
    print(r["OP CODE"], r["DEVICE KERNEL DURATION [ns]"], r["DEVICE FW DURATION [ns]"])
