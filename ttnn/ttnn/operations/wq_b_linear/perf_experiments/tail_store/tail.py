"""Tail analysis: per dispatch span, plus for dispatches 2..N the median over dispatches of
(max over workers) end of each RISC's kernel and of zones; and the last-ending core breakdown. Usage: tail.py csv"""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1.35
rows = list(csv.reader(open(sys.argv[1])))[2:]
runs = sorted({int(r[7]) for r in rows})
res = []
for run in runs:
    rr = [r for r in rows if int(r[7]) == run]
    ks = [int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL")]
    med = sorted(ks)[len(ks) // 2]
    ks = [t for t in ks if abs(t - med) < 1e6]
    t0 = min(ks)
    span = (max(ks) - t0) / FREQ
    ends = defaultdict(dict)  # name/risc -> core -> end
    for r in rr:
        t = int(r[5])
        if abs(t - med) > 1e6 or r[11].strip() != "ZONE_END":
            continue
        name = r[10].strip()
        if name.endswith("-FW"):
            continue
        key = name if not name.endswith("-KERNEL") else r[3].strip() + "-K"
        core = (int(r[1]), int(r[2]))
        ends[key][core] = max(ends[key].get(core, 0), (t - t0) / FREQ)
    res.append((span, ends))
print("spans:", [round(s) for s, _ in res])
use = res[1:] if len(res) > 1 else res
print("median span (excl first):", round(st.median(s for s, _ in use)))
keys = sorted({k for _, e in use for k in e})
print(f"{'key':24s} {'max_end_med':>11s} {'med_end_med':>11s}")
for k in sorted(keys, key=lambda k: st.median(max(e[k].values()) for _, e in use if k in e)):
    mx = st.median(max(e[k].values()) for _, e in use if k in e)
    md = st.median(st.median(e[k].values()) for _, e in use if k in e)
    print(f"{k:24s} {mx:11.0f} {md:11.0f}")
# last core of last dispatch
span, e = res[-1]
lastcore = max(e["BRISC-K"], key=e["BRISC-K"].get)
print("last-dispatch last BRISC core", lastcore, {k: round(v[lastcore]) for k, v in e.items() if lastcore in v})
