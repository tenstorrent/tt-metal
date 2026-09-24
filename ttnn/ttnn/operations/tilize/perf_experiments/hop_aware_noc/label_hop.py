"""Pair a --profile report's generic-op rows with the hop harness' P2 labels; medians + delta vs the first variant.
usage (repo root): python3 label_hop.py <pytest -s log> [report_dir]; pass the report dir the run printed
(SAFE_PYTEST: PROFILER CSV) when other sessions share the device, the default is the newest. low_l1 cases run the op twice (DOUBLE)."""
import csv, glob, sys
from collections import defaultdict

log = sys.argv[1]
rep = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("generated/profiler/reports/*/"))[-1]
DOUBLE = {"l1_lowl1"}  # low_l1 scenarios run the op at both settings (A/B)
rows = [
    r
    for r in csv.DictReader(open(glob.glob(rep + "ops_perf_results*.csv")[0]))
    if "generic" in r["OP CODE"].lower() and r["DEVICE KERNEL DURATION [ns]"]
]
labels = [l.split()[1:3] for l in open(log) if l.startswith("P2 ")]
exp = []
for c, v in labels:
    c = c.split("=")[1]
    v = v.split("=")[1].split("kernels_")[-1]
    if c in DOUBLE:
        exp += [(c + "_off", v), (c + "_on", v)]
    else:
        exp.append((c, v))
exp = exp[-len(rows) :]
assert len(exp) == len(rows), (len(exp), len(rows))
d = defaultdict(list)
for k, r in zip(exp, rows):
    d[k].append(int(float(r["DEVICE KERNEL DURATION [ns]"])))
cases = list(dict.fromkeys(k[0] for k in d))
vs = list(dict.fromkeys(k[1] for k in d))
for c in cases:
    meds = {v: sorted(d[(c, v)])[len(d[(c, v)]) // 2] for v in vs if (c, v) in d}
    h = meds.get(vs[0])
    print(
        f"{c:>12s} "
        + "  ".join(
            f"{v} {meds[v]:>6d} {str(d[(c,v)]):<22s}" + (f" {100*(meds[v]-h)/h:+5.1f}%" if v != vs[0] and h else "")
            for v in vs
            if v in meds
        )
    )
