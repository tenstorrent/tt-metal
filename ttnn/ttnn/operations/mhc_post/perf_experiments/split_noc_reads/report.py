"""Match the LAST len(run_order) GenericOp rows of an ops_perf_results CSV to run_order.jsonl and tabulate
DEVICE KERNEL DURATION [ns] (column 20) per (shape, mode) x variant, with ratio vs `orig`.
usage: python report.py <ops_perf_results.csv> [run_order.jsonl]"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

csv_path = sys.argv[1]
order_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "run_order.jsonl"
order = [json.loads(l) for l in open(order_path) if l.strip()]
with open(csv_path) as f:
    rows = [r for r in csv.reader(f)][1:]
rows = [r for r in rows if r[0].startswith("GenericOp")]
rows = rows[-len(order) :]
assert len(rows) == len(order), (len(rows), len(order))
tab = defaultdict(dict)
variants = []
for o, r in zip(order, rows):
    tab[(o["shape"], o["mode"])][o["variant"]] = int(r[19])
    if o["variant"] not in variants:
        variants.append(o["variant"])
print(f"{'shape':28s} {'mode':8s} " + " ".join(f"{v:>14s}" for v in variants))
for (shape, mode), d in tab.items():
    base = d.get("orig")
    cells = []
    for v in variants:
        if v not in d:
            cells.append(f"{'-':>14s}")
        elif base:
            cells.append(f"{d[v]/1000:7.1f}({d[v]/base:4.2f})")
        else:
            cells.append(f"{d[v]/1000:14.1f}")
    print(f"{shape:28s} {mode:8s} " + " ".join(cells))
