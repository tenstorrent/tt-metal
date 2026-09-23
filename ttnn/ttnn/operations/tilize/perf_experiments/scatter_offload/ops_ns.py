"""Print DEVICE KERNEL DURATION [ns] per op row of a --profile report, labelled by a name list.

usage: python ops_ns.py "name1,name2,..." [report_dir]  (default: newest report). Rows of the
generic op (tilize) are mapped to names in collection order.
"""
import csv, glob, os, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../..")
names = sys.argv[1].split(",")
rep = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(root, "generated/profiler/reports/*/")))[-1]
rows = list(csv.DictReader(open(glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0])))
rows = [
    r
    for r in rows
    if r["DEVICE KERNEL DURATION [ns]"] and "Generic" in r["OP CODE"] or "generic" in r["OP CODE"].lower()
]
print(f"report {rep}: {len(rows)} op rows, {len(names)} names")
for n, r in zip(names, rows):
    print(f"{n:28s} {r['DEVICE KERNEL DURATION [ns]']:>8s}  cores={r['CORE COUNT']}")
