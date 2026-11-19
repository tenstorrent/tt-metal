# usage: kgrid.py <csv> <run_index> [risc]: kernel start (us after the run's earliest) per physical core, as a grid.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
risc = sys.argv[3] if len(sys.argv) > 3 else "BRISC"
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
ev = runs[big[idx - 1]]
ks = {}
for core, rk, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_START" and rk == risc:
        ks[core] = min(ks.get(core, t), t)
t0 = min(ks.values())
xs = sorted({c[0] for c in ks})
ys = sorted({c[1] for c in ks})
print("y\\x " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"{y:3d} " + " ".join(f"{(ks[(x,y)]-t0)/freq:5.2f}" if (x, y) in ks else "    ." for x in xs))
