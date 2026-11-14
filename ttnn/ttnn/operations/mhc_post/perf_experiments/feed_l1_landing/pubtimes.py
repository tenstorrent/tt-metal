"""w_publish end per worker (us from kernel start), grouped by physical core row y; usage: pubtimes.py csv dev"""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1350.0
path, want = sys.argv[1], int(sys.argv[2])
rows = []
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    rows = [x for x in r if int(x[0]) == want]
runs = sorted({int(x[7]) for x in rows})
run = max(r for r in runs if sum(1 for x in rows if int(x[7]) == r and x[10].endswith("KERNEL")) > 200)
rows = [x for x in rows if int(x[7]) == run]
t0 = min(int(x[5]) for x in rows if x[10].endswith("KERNEL") and x[11] == "ZONE_START")
ends = {}
for x in rows:
    if x[10].strip() == "w_publish" and x[11] == "ZONE_END":
        ends[(int(x[1]), int(x[2]))] = (int(x[5]) - t0) / FREQ
by_y = defaultdict(list)
by_x = defaultdict(list)
for (cx, cy), t in ends.items():
    by_y[cy].append(t)
    by_x[cx].append(t)
for y in sorted(by_y):
    v = by_y[y]
    print(f"y={y:2d} n={len(v):3d} min {min(v):6.1f} med {st.median(v):6.1f} max {max(v):6.1f}")
for x in sorted(by_x):
    v = by_x[x]
    print(f"x={x:2d} n={len(v):3d} min {min(v):6.1f} med {st.median(v):6.1f} max {max(v):6.1f}")
