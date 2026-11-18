# usage: coregrid.py <profile_log_device.csv> <run index (1-based among runs with >= 8 cores)> <zone> [risc]
# Per-core grid (x across, y down) of a zone's summed duration and its end time (us from the run's first kernel start).
import collections, csv, sys

path, idx, zone = sys.argv[1], int(sys.argv[2]), sys.argv[3]
risc_want = sys.argv[4] if len(sys.argv) > 4 else "TRISC_1"
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
dur = collections.defaultdict(float)
end = {}
st = {}
open_ = {}
for core, risc, t, z, kind in ev:
    if risc != risc_want or z != zone:
        continue
    if kind == "ZONE_START":
        open_[core] = t
        st.setdefault(core, t)
    elif core in open_:
        dur[core] += (t - open_.pop(core)) / freq
        end[core] = (t - t0) / freq
xs = sorted({c[0] for c in dur})
ys = sorted({c[1] for c in dur})
print(f"{zone} on {risc_want}: duration (us) / end (us)")
print("     " + "".join(f"{x:>12}" for x in xs))
for y in ys:
    print(f"{y:>4} " + "".join(f"{dur.get((x, y), 0):5.1f}/{end.get((x, y), 0):5.1f} " for x in xs))
