# Per compute core: median compute_normalize zone duration (us) per RISC, and xg_ahead duration, over calls 2..
import csv, sys, collections, statistics as st

F = 1350.0
z = collections.defaultdict(lambda: collections.defaultdict(list))
with open(sys.argv[1]) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for x in r:
        name, typ = x[10], x[11]
        if name not in ("compute_normalize", "compute_xg_ahead", "compute_combine"):
            continue
        z[(x[0], x[1], x[2], x[3], name, int(x[7]))][typ].append(int(x[5]))
d = collections.defaultdict(list)
for (chip, cx, cy, risc, name, run), v in z.items():
    if v["ZONE_START"] and v["ZONE_END"]:
        d[(chip, cx, cy, risc, name)].append((max(v["ZONE_END"]) - min(v["ZONE_START"])) / F)
for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
    for name in ("compute_xg_ahead", "compute_normalize"):
        vals = sorted(st.median(v[1:]) for k, v in d.items() if k[3] == risc and k[4] == name and len(v) > 1)
        if vals:
            q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
            print(
                f"{risc} {name:18s} n={len(vals)} min={vals[0]:.2f} p25={q(.25):.2f} p50={q(.5):.2f} p75={q(.75):.2f} max={vals[-1]:.2f}"
            )
