"""Span + x-reader timings per dispatch (median of dispatches 2..6). Usage: analyze.py csv
reads_done = end of x_read_drain (sender NCRISC); r0_push = end of the first x_mcast_wait (sender BRISC sees round 0);
mcast_done = end of x_mcast_barrier. Times in ns after the dispatch's first kernel start."""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1.35
rows = list(csv.reader(open(sys.argv[1])))[2:]
runs = sorted({int(r[7]) for r in rows})
res = defaultdict(list)
for run in runs:
    rr = [r for r in rows if int(r[7]) == run]
    ks = [int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL")]
    med = sorted(ks)[len(ks) // 2]
    ks = [t for t in ks if abs(t - med) < 1e6]
    t0 = min(ks)
    res["span"].append((max(ks) - t0) / FREQ)
    first_end = {}
    last_end = defaultdict(int)
    for r in rr:
        name, t = r[10].strip(), int(r[5])
        if r[11].strip() != "ZONE_END" or abs(t - med) > 1e6:
            continue
        first_end.setdefault(name, t)
        first_end[name] = min(first_end[name], t)
        last_end[name] = max(last_end[name], t)
    for key, (zone, which) in {
        "r0_push": ("x_mcast_wait", first_end),
        "reads_done": ("x_read_drain", last_end),
        "issue_end": ("x_read_issue", last_end),
        "mcast_done": ("x_mcast_barrier", last_end),
    }.items():
        if zone in which:
            res[key].append((which[zone] - t0) / FREQ)
use = slice(1, None) if len(runs) > 1 else slice(None)
print("spans:", [round(s) for s in res["span"]])
print("  ".join(f"{k}={round(st.median(v[use]))}" for k, v in res.items()))
