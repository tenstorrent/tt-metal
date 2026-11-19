# usage: zone_grid.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> <zone> [end]
# One zone's per-core duration (or with `end`: its end time relative to the run's first kernel start), in us, laid out
# on the virtual core grid (rows = y, columns = x).
import collections, csv, sys

path, idx, zone = sys.argv[1], int(sys.argv[2]), sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "dur"
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL"))
val = collections.defaultdict(float)
st = {}
for core, risc, t, z, kind in ev:
    if z != zone:
        continue
    if kind == "ZONE_START":
        st[(core, risc)] = t
    elif kind == "ZONE_END" and (core, risc) in st:
        if mode == "end":
            val[core] = max(val[core], (t - t0) / freq)
        else:
            val[core] += (t - st[(core, risc)]) / freq
xs = sorted({c[0] for c in val})
ys = sorted({c[1] for c in val})
print(f"{zone} ({mode}, us) x: " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"y={y:2d}            " + " ".join(f"{val[(x, y)]:5.1f}" if (x, y) in val else "    ." for x in xs))
