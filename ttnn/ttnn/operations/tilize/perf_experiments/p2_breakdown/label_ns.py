"""Pair the generic-op rows of a --profile report with the harness' `P2 case=.. variant=..` lines.

usage: python label_ns.py <pytest -s log> [report_dir]   (default: newest report)
The precompile pass prints fake-device P2 lines first, so the LAST n lines (n = op rows) are used.
"""
import csv, glob, os, re, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../..")
rep = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(root, "generated/profiler/reports/*/")))[-1]
rows = list(csv.DictReader(open(glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0])))
rows = [r for r in rows if "generic" in r["OP CODE"].lower() and r["DEVICE KERNEL DURATION [ns]"]]
labels = [l.split()[1:3] for l in open(sys.argv[1]) if l.startswith("P2 ")][-len(rows) :]
out = {}
for (c, v), r in zip(labels, rows):
    c = c.split("=")[1]
    v = v.split("=")[1].split("kernels_")[-1]
    out.setdefault(c, {})[v] = int(float(r["DEVICE KERNEL DURATION [ns]"]))
vs = list(dict.fromkeys(v for d in out.values() for v in d))
print("case " + " ".join(f"{v:>7s}" for v in vs))
for c, d in out.items():
    print(f"{c:>4s} " + " ".join(f"{d.get(v, 0):>7d}" for v in vs))
