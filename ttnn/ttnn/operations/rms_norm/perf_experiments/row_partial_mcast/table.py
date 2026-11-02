"""table.py <tag>: DEVICE KERNEL DURATION per (cell, variant) from runs/<tag>_ops.csv + runs/<tag>_labels.txt."""
import csv, sys, collections
from pathlib import Path

D = Path(__file__).parent / "runs"
T = sys.argv[1]
labels = [l.strip() for l in open(D / f"{T}_labels.txt") if l.strip()]
RUNS_ROWS = None
rows = [r for r in csv.DictReader(open(D / f"{T}_ops.csv")) if r["OP CODE"].startswith("GenericOp")]
# helper programs (e.g. fill_implicit_tile_padding) are not GenericOp; rms_norm is one GenericOp per variant
# the up-front-collect pass runs every test once without a device first: keep the LAST len(rows) labels
labels = labels[-len(rows) :]
tab = collections.OrderedDict()
for lab, r in zip(labels, rows):
    cell, var = lab.split(":")
    tab.setdefault(cell, {})[var] = int(float(r["DEVICE KERNEL DURATION [ns]"]))
vars_ = []
for c in tab.values():
    for v in c:
        if v not in vars_:
            vars_.append(v)
print("cell".ljust(28) + "".join(v.rjust(12) for v in vars_) + "   speedup(base/x)")
for cell, d in tab.items():
    line = cell.ljust(28) + "".join(str(d.get(v, "-")).rjust(12) for v in vars_)
    if "base" in d:
        line += "   " + " ".join(f"{v}={d['base'] / d[v]:.2f}" for v in vars_ if v != "base" and v in d)
    print(line)
