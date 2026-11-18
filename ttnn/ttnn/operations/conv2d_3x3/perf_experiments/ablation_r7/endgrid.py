# usage: endgrid.py <csv> <run_index>: per-core latest kernel end (us, from the run's first kernel start) as a grid.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
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
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = runs[big[idx - 1]]
st = min(t for c, z, k, t in ev if z.endswith("-KERNEL") and k == "ZONE_START")
end = collections.defaultdict(int)
for c, z, k, t in ev:
    if z.endswith("-KERNEL") and k == "ZONE_END":
        end[c] = max(end[c], t)
xs = sorted({c[0] for c in end})
ys = sorted({c[1] for c in end})
print("y\\x " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"{y:3d} " + " ".join(f"{(end[(x,y)]-st)/freq:5.1f}" if (x, y) in end else "    -" for x in xs))
