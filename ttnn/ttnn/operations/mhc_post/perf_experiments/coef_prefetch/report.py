"""Match cpp_device_perf_report.csv rows (call order) to the LAST pid's entries of run_order.jsonl; print
DEVICE KERNEL DURATION per shape x variant (median over reps, all reps listed) and ratio vs `base`.
usage: python report.py <cpp_device_perf_report.csv> [run_order.jsonl]"""
import csv, json, sys, statistics
from collections import defaultdict
from pathlib import Path

csv_path = sys.argv[1]
order_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "run_order.jsonl"
order = [json.loads(l) for l in open(order_path) if l.strip()]
pid = order[-1]["pid"]
order = [o for o in order if o["pid"] == pid]
rows = list(csv.DictReader(open(csv_path)))[-len(order) :]
assert len(rows) == len(order), (len(rows), len(order))
tab = defaultdict(lambda: defaultdict(list))
variants = []
for o, r in zip(order, rows):
    tab[o["shape"]][o["variant"]].append(int(r["DEVICE KERNEL DURATION [ns]"]))
    if o["variant"] not in variants:
        variants.append(o["variant"])
print(f"{'shape':24s} " + " ".join(f"{v:>22s}" for v in variants))
for shape, d in tab.items():
    base = statistics.median(d["base"]) if d.get("base") else None
    cells = []
    for v in variants:
        if v not in d:
            cells.append(f"{'-':>22s}")
            continue
        m = statistics.median(d[v])
        reps = "/".join(f"{x/1000:.1f}" for x in d[v])
        cells.append(f"{m/1000:6.1f}({m/base:4.3f})[{reps}]" if base else f"{m/1000:6.1f}[{reps}]")
    print(f"{shape:24s} " + " ".join(cells))
print(
    "runs (GLOBAL CALL COUNT):",
    " ".join(f"{o['shape']}/{o['variant']}/{o['rep']}={r['GLOBAL CALL COUNT']}" for o, r in zip(order, rows)),
)
