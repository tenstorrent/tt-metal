# usage: percore.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [zone ...]
# Per core (physical x, y): first start / last end (us after the run's earliest kernel start) of each named zone
# (default: reader_fill, writer_fill, compute tilize, matmul walk, kernel end), sorted by kernel end.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
zones = sys.argv[3:] or [
    "reader_fill",
    "writer_fill",
    "reader_center",
    "compute_tilize",
    "compute_matmul_walk",
]
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
st = {}
en = {}
for core, risc, t, z, kind in ev:
    key = (core, z if not z.endswith("-KERNEL") else "KERNEL")
    if kind == "ZONE_START":
        st[key] = min(st.get(key, t), t)
    else:
        en[key] = max(en.get(key, t), t)
cores = sorted({c for c, z in st})
us = lambda v: (v - t0) / freq
print("core     " + " ".join(f"{z[:18]:>14}" for z in zones) + "   kend")
for c in sorted(cores, key=lambda c: en.get((c, "KERNEL"), 0)):
    cells = []
    for z in zones:
        if (c, z) in st:
            cells.append(f"{us(st[(c, z)]):5.1f}-{us(en[(c, z)]):5.1f} ")
        else:
            cells.append(" " * 14)
    print(f"{str(c):8s} " + " ".join(f"{x:>14}" for x in cells) + f"  {us(en[(c, 'KERNEL')]):5.1f}")
