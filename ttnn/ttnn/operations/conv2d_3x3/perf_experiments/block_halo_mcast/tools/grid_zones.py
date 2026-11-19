# usage: grid_zones.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> <zone> [risc] [start|end|dur]
# Prints a y-by-x grid of one zone's end (default) time in us relative to the run's earliest kernel start.
import collections, csv, sys

path, idx, zone = sys.argv[1], int(sys.argv[2]), sys.argv[3]
risc_want = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
what = sys.argv[5] if len(sys.argv) > 5 else "end"
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
st, en = {}, {}
for core, risc, t, z, kind in ev:
    if z != zone or (risc_want and risc != risc_want):
        continue
    if kind == "ZONE_START":
        st.setdefault(core, (t - t0) / freq)
    else:
        en[core] = (t - t0) / freq
xs = sorted({c[0] for c in st})
ys = sorted({c[1] for c in st})
print(f"run {big[idx-1]} zone {zone} {what}")
print("y\\x " + " ".join(f"{x:6d}" for x in xs))
for y in ys:
    vals = []
    for x in xs:
        if (x, y) in st and (x, y) in en:
            v = {"start": st[(x, y)], "end": en[(x, y)], "dur": en[(x, y)] - st[(x, y)]}[what]
            vals.append(f"{v:6.2f}")
        else:
            vals.append("     -")
    print(f"{y:3d} " + " ".join(vals))
