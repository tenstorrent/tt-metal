"""Per-core start-up chain from a profile_log_device.csv (last big run of one device).
usage: python percore.py <csv> [device] ; prints, per core, the first occurrence (start, end) of each zone (us)."""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1350.0
path, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    xs = [x for x in r if int(x[0]) == dev]
runs = sorted({int(x[7]) for x in xs})
run = max(q for q in runs if sum(1 for x in xs if int(x[7]) == q and x[10].endswith("KERNEL")) > 200)
xs = [x for x in xs if int(x[7]) == run]
t0 = min(int(x[5]) for x in xs if x[10].endswith("KERNEL") and x[11] == "ZONE_START")
open_, first = {}, defaultdict(dict)
for x in sorted(xs, key=lambda x: int(x[5])):
    key = ((int(x[1]), int(x[2])), x[3], x[10])
    t = (int(x[5]) - t0) / FREQ
    if x[11] == "ZONE_START":
        open_.setdefault(key, []).append(t)
    elif x[11] == "ZONE_END" and open_.get(key):
        s = open_[key].pop()
        first[key[0]].setdefault(x[10], (s, t))
names = sorted({n for d in first.values() for n in d if "FW" not in n and "KERNEL" not in n})
want = [n for n in sys.argv[3].split(",")] if len(sys.argv) > 3 else names
print("core     " + "".join(f"{n:>22s}" for n in want))
rows = sorted(first.items(), key=lambda kv: kv[1].get(want[0], (0, 0))[1])
for c, d in rows:
    print(f"{str(c):9s}" + "".join(f"{d[n][0]:10.1f}->{d[n][1]:6.1f}  " if n in d else f"{'-':>22s}" for n in want))
