"""Map ops CSVs (one GenericOp row per test_pw item, collection order) to a case x variant table.
usage: python table.py <collected ids file (pytest --collect-only -q)> <ops_perf_results.csv> [more csvs...]
With several CSVs (repeat runs) prints the per-cell median and (min..max)."""
import csv, re, sys, statistics
from collections import defaultdict

ids = [l.strip() for l in open(sys.argv[1]) if "test_pw[" in l]
tab = defaultdict(lambda: defaultdict(list))
vs = []
for f in sys.argv[2:]:
    rows = [r for r in csv.DictReader(open(f)) if r["OP CODE"] == "GenericOpDeviceOperation"]
    assert len(rows) == len(ids), (f, len(rows), len(ids))
    for r, i in zip(rows, ids):
        c, v = re.search(r"case=([^-\]]+)-variant=([^\]]+)", i).groups()
        tab[c][v].append(int(r["DEVICE KERNEL DURATION [ns]"]))
        if v not in vs:
            vs.append(v)
print(f"median of {len(sys.argv) - 2} run(s); (ratio vs baseline median) [min..max]")
print(f"{'case':16s}" + "".join(f"{v:>26s}" for v in vs))
for c, d in tab.items():
    b = statistics.median(d["baseline"]) if "baseline" in d else None
    cells = []
    for v in vs:
        m = statistics.median(d[v])
        cells.append(f"{m:>7.0f}({m/b:4.2f})[{min(d[v])}..{max(d[v])}]" if b else f"{m:>26.0f}")
    print(f"{c:16s}" + "".join(f"{x:>26s}" for x in cells))
