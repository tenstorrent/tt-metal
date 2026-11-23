"""Multicast window per dispatch: first x_mcast_issue start -> x_mcast_barrier end (sender BRISC), ns.
Also first-issue start and barrier end relative to op start. Usage: mcast.py csv -> median over dispatches 2..N."""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1.35
rows = list(csv.reader(open(sys.argv[1])))[2:]
by = defaultdict(list)
for r in rows:
    by[int(r[7])].append(r)
win, s0, e0 = [], [], []
for run in sorted(by)[1:]:
    rr = by[run]
    ks = [int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL")]
    med = sorted(ks)[len(ks) // 2]
    t0 = min(t for t in ks if abs(t - med) < 1e6)
    iss = [int(r[5]) for r in rr if r[10].strip() == "x_mcast_issue" and r[11].strip() == "ZONE_START"]
    bar = [int(r[5]) for r in rr if r[10].strip() == "x_mcast_barrier" and r[11].strip() == "ZONE_END"]
    win.append((max(bar) - min(iss)) / FREQ)
    s0.append((min(iss) - t0) / FREQ)
    e0.append((max(bar) - t0) / FREQ)
print(f"mcast window med {st.median(win):.0f} ns (first issue at {st.median(s0):.0f}, done at {st.median(e0):.0f})")
