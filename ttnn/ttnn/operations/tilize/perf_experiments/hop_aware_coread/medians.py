"""Median DEVICE KERNEL DURATION [ns] per (case, variant) over several label_ns.py-parsable logs.

usage: python medians.py <log> [<log> ...]   (each log = one --profile session; its report is the
newest report created before the next log's, so pass the report dirs explicitly with REPORTS=a,b,c)
"""
import os, statistics, subprocess, sys

LABEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../p2_breakdown/label_ns.py")
reps = os.environ.get("REPORTS", "").split(",") if os.environ.get("REPORTS") else [None] * (len(sys.argv) - 1)
data = {}
for log, rep in zip(sys.argv[1:], reps):
    out = subprocess.run([sys.executable, LABEL, log] + ([rep] if rep else []), capture_output=True, text=True).stdout
    lines = out.strip().splitlines()
    vs = lines[0].split()[1:]
    for l in lines[1:]:
        c, *ns = l.split()
        for v, n in zip(vs, ns):
            data.setdefault(c, {}).setdefault(v, []).append(int(n))
vs = list(dict.fromkeys(v for d in data.values() for v in d))
base = os.environ.get("BASE", "head")
print("median ns (n runs); % vs " + base)
print(f"{'case':>6s} " + " ".join(f"{v:>14s}" for v in vs))
for c, d in data.items():
    b = statistics.median(d[base]) if base in d else None
    cells = []
    for v in vs:
        m = statistics.median(d[v])
        cells.append(f"{int(m):>7d}({len(d[v])})" + (f"{100 * (m / b - 1):+5.1f}%" if b else ""))
    print(f"{c:>6s} " + " ".join(f"{x:>14s}" for x in cells))
