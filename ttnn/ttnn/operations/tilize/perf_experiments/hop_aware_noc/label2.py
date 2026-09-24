"""Pair a --profile report's generic-op rows with the hop harness' P2 labels; per (case, variant):
median, n, runs, delta vs the first variant. usage (repo root):
  python3 label2.py <pytest -s log> <report_dir printed as "SAFE_PYTEST: PROFILER CSV">
low_l1 cases run the op twice (DOUBLE: both rows labelled _off / _on)."""
import csv, glob, sys
from collections import defaultdict

log, rep = sys.argv[1], sys.argv[2]
DOUBLE = {"l1_lowl1"}
rows = [
    r
    for r in csv.DictReader(open(glob.glob(rep.rstrip("/") + "/ops_perf_results*.csv")[0]))
    if "generic" in r["OP CODE"].lower() and r["DEVICE KERNEL DURATION [ns]"]
]
labels = [l.split()[1:3] for l in open(log) if l.startswith("P2 ")]
exp = []
for c, v in labels:
    c = c.split("=", 1)[1]
    v = v.split("=", 1)[1]
    exp += [(c + "_off", v), (c + "_on", v)] if c in DOUBLE else [(c, v)]
exp = exp[-len(rows) :]
assert len(exp) == len(rows), (len(exp), len(rows))
d = defaultdict(list)
for k, r in zip(exp, rows):
    d[k].append(int(float(r["DEVICE KERNEL DURATION [ns]"])))
cases = list(dict.fromkeys(k[0] for k in d))
vs = list(dict.fromkeys(k[1] for k in d))
med = lambda x: sorted(x)[len(x) // 2] if len(x) % 2 else (sorted(x)[len(x) // 2 - 1] + sorted(x)[len(x) // 2]) / 2
print("case | " + " | ".join(vs))
for c in cases:
    h = med(d[(c, vs[0])]) if (c, vs[0]) in d else None
    cells = []
    for v in vs:
        if (c, v) not in d:
            cells.append("-")
            continue
        m = med(d[(c, v)])
        cells.append(
            f"{m:.0f} (n={len(d[(c, v)])}{'' if v == vs[0] or not h else f', {100 * (m - h) / h:+.1f}%'}) {d[(c, v)]}"
        )
    print(f"{c} | " + " | ".join(cells))
