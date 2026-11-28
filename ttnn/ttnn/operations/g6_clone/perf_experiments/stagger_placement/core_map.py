"""Per-core corrected start and end (cycles after the earliest corrected start of the launch), median over
launches after the first, in physical (profiler) coordinates. Usage: core_map.py <profile_log_device.csv>"""
import collections, csv, json, statistics, sys
from pathlib import Path

OFF = json.load(open(Path(__file__).parent / "clock_offset_map.json"))["offset"]
start = collections.defaultdict(dict)
end = collections.defaultdict(dict)
with open(sys.argv[1]) as f:
    next(f)
    for r in csv.DictReader(f, skipinitialspace=True):
        if not r["zone name"].endswith("-KERNEL"):
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
        run = int(r["run host ID"])
        t = int(r["time[cycles since reset]"]) - OFF[f"{int(r['core_x'])},{int(r['core_y'])}"]
        if r["type"] == "ZONE_START":
            start[run][c] = min(t, start[run].get(c, t))
        else:
            end[run][c] = max(t, end[run].get(c, t))
runs = sorted(start)[1:]
S = collections.defaultdict(list)
E = collections.defaultdict(list)
for run in runs:
    t0 = min(start[run].values())
    for c in start[run]:
        S[c].append(start[run][c] - t0)
        E[c].append(end[run][c] - t0)
xs = sorted({c[0] for c in S})
ys = sorted({c[1] for c in S})
for name, D in (("corrected start", S), ("corrected end", E)):
    print(name)
    print("y\\x " + " ".join(f"{x:>5}" for x in xs))
    for y in ys:
        print(f"{y:>3} " + " ".join(f"{statistics.median(D[(x,y)]):5.0f}" if (x, y) in D else "    ." for x in xs))
