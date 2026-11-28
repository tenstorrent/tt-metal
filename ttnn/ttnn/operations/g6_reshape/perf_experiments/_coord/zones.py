"""zones.py <profile_log_device.csv>: per-zone duration (ns) medians over cores and launches 2..N, per RISC,
plus zone start offsets relative to that core's kernel start."""
import csv, sys, statistics, collections

p = sys.argv[1]
f = open(p)
next(f)
rows = list(csv.DictReader(f, skipinitialspace=True))
FREQ = 1.35
starts = {}
dur = collections.defaultdict(list)
off = collections.defaultdict(list)
kstart = {}
for r in rows:
    key = (r["core_x"], r["core_y"], r["RISC processor type"], r["run host ID"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        starts[key] = t
        if r["zone name"].endswith("-KERNEL"):
            kstart[key[:4]] = t
    elif r["type"] == "ZONE_END" and key in starts:
        if r["run host ID"] == min(x["run host ID"] for x in rows[:1]):
            pass
        dur[(r["RISC processor type"], r["zone name"])].append(
            (int(r["run host ID"]), (t - starts[key]) / FREQ, (starts[key] - kstart.get(key[:4], starts[key])) / FREQ)
        )
first = min(v[0] for vs in dur.values() for v in vs)
for k in sorted(dur):
    v = [x for x in dur[k] if x[0] != first]
    d = [x[1] for x in v]
    o = [x[2] for x in v]
    print(
        f"{k[0]:7s} {k[1]:14s} n={len(d):5d} med={statistics.median(d):7.0f} max={max(d):7.0f} start_off_med={statistics.median(o):6.0f}"
    )
