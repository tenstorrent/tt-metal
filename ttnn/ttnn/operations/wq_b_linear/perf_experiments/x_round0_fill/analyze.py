"""Span (median of dispatches 2..6) + x round 0 arrival for the x_round0_fill bench. Usage: analyze.py csv"""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1.35
rows = list(csv.reader(open(sys.argv[1])))[2:]
runs = sorted({int(r[7]) for r in rows})
spans, x0, cw, rd = [], [], [], []
for run in runs:
    rr = [r for r in rows if int(r[7]) == run]
    ks = [int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL")]
    med = sorted(ks)[len(ks) // 2]
    ks = [t for t in ks if abs(t - med) < 1e6]
    t0 = min(ks)
    spans.append((max(ks) - t0) / FREQ)
    ends = defaultdict(dict)
    for r in rr:
        name = r[10].strip()
        if r[11].strip() == "ZONE_END" and abs(int(r[5]) - med) < 1e6:
            ends[name][(r[1], r[2], r[3])] = (int(r[5]) - t0) / FREQ
    x0.append(ends["x_wait_r0"].values())
    cw.append([v for k, v in ends["compute_wait_x0"].items() if k[2].strip() == "TRISC_0"])
    rd.append(ends.get("x_mcast_issue", {}).values())
use = slice(1, None) if len(runs) > 1 else slice(None)
print("spans:", [round(s) for s in spans], "median:", round(st.median(spans[use])))
x0m = [st.median(v) for v in x0][use]
x0x = [max(v) for v in x0][use]
cwm = [st.median(v) for v in cw][use]
cwx = [max(v) for v in cw][use]
print(
    "x_r0 land med/max:",
    round(st.median(x0m)),
    round(st.median(x0x)),
    " compute start (TRISC_0) med/max:",
    round(st.median(cwm)),
    round(st.median(cwx)),
)
