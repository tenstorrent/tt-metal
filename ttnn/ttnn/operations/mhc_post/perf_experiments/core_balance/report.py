"""Match the LAST len(run_order) rows of cpp_device_perf_report.csv to run_order_<session>.jsonl and tabulate
DEVICE KERNEL DURATION [ns] per shape x variant (median over reps), ratio vs `orig`.
usage: python3 report.py <cpp_device_perf_report.csv> <session>"""
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

csv_path, session = sys.argv[1], sys.argv[2]
order = [json.loads(l) for l in open(Path(__file__).parent / f"run_order_{session}.jsonl") if l.strip()]
with open(csv_path) as f:
    rd = csv.reader(f)
    hdr = next(rd)
    rows = list(rd)
col = hdr.index("DEVICE KERNEL DURATION [ns]")
rows = rows[-len(order) :]
assert len(rows) == len(order), (len(rows), len(order))
tab = defaultdict(lambda: defaultdict(list))
variants = []
for o, r in zip(order, rows):
    tab[o["shape"]][o["variant"]].append(int(r[col]))
    if o["variant"] not in variants:
        variants.append(o["variant"])
print(f"{'shape':26s} " + " ".join(f"{v:>22s}" for v in variants))
for shape, d in tab.items():
    base = statistics.median(d["orig"]) if d.get("orig") else None
    cells = []
    for v in variants:
        if v not in d:
            cells.append(f"{'-':>22s}")
            continue
        m = statistics.median(d[v])
        allv = "/".join(f"{x/1000:.1f}" for x in d[v])
        s = f"{m/1000:.1f}" + (f"({m/base:.3f})" if base else "")
        cells.append(f"{s:>14s} [{allv}]" if len(d[v]) > 1 else f"{s:>22s}")
    print(f"{shape:26s} " + " ".join(cells))
