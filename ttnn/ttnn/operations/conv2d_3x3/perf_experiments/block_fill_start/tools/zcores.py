# usage: zcores.py <csv> <run idx 1-based among runs with >= 8 cores> <zone>[,<zone>...] [risc]
# Per core: first start / last end (us after the run's earliest kernel start) of each named zone, sorted by end.
import collections, csv, sys

path, idx, zones = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
want_risc = sys.argv[4] if len(sys.argv) > 4 else None
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
first, last, tot = {}, {}, collections.defaultdict(float)
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if z not in zones or (want_risc and risc != want_risc):
        continue
    key = (core, risc, z)
    if kind == "ZONE_START":
        stack[key].append(t)
        first[key] = min(first.get(key, t), t)
    elif stack[key]:
        s = stack[key].pop()
        last[key] = max(last.get(key, t), t)
        tot[key] += (t - s) / freq
for z in zones:
    ks = sorted([k for k in last if k[2] == z], key=lambda k: last[k])
    print(f"== {z}: {len(ks)} (core,risc)")
    line = []
    for k in ks:
        line.append(f"{k[0][0]},{k[0][1]}/{k[1][:2]}:{us(first[k]):.2f}-{us(last[k]):.2f}")
    for i in range(0, len(line), 6):
        print("   " + "  ".join(line[i : i + 6]))
