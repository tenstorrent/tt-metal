"""Zip ops_perf_results CSV rows (DEVICE KERNEL DURATION [ns]) with run_labels.jsonl (same dispatch order).

usage: python parse_perf.py <ops_perf_results.csv> [labels.jsonl]   (host-only, no device)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv_path = sys.argv[1]
labels_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "run_labels.jsonl")
with open(csv_path) as f:
    rows = list(csv.reader(f))
hdr = rows[0]
col = hdr.index("DEVICE KERNEL DURATION [ns]")
opc = hdr.index("OP CODE")
ops = [r for r in rows[1:] if "generic" in r[opc].lower() or "Generic" in r[opc]]
labels = [json.loads(l) for l in open(labels_path)]
if len(ops) != len(labels):
    print(f"WARNING: {len(ops)} generic-op rows vs {len(labels)} labels", file=sys.stderr)
by_shape = defaultdict(dict)
for r, lab in zip(ops, labels):
    shape, var = lab["label"].split("|")
    extra = {k: v for k, v in lab.items() if k != "label"}
    by_shape[shape].setdefault(var, ([], extra))[0].append(float(r[col]))
import statistics

med = statistics.median
for shape, vs in by_shape.items():
    base = med(vs["baseline"][0]) if "baseline" in vs else None
    print(shape)
    for var, (nss, extra) in vs.items():
        ns = med(nss)
        rel = f"  {base / ns:5.3f}x" if base else ""
        runs = " [" + " ".join(f"{x / 1000:.1f}" for x in nss) + "]" if len(nss) > 1 else ""
        print(f"   {var:12s} {ns / 1000:9.1f} us{rel}{runs}  {extra if extra else ''}")
