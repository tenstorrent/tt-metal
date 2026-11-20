# usage: corestat.py <profile_log_device.csv> <run_index> : per core (grid map), round-0 ready seen (writer_r0_ready),
# the compute weight-wait end, the first slab-wait end (unnamed / compute_slab_wait zone on TRISC_0) and the TRISC_0
# kernel end, us after the run's earliest kernel start.
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
st = collections.defaultdict(list)
span = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    key = (core, risc, z)
    if kind == "ZONE_START":
        st[key].append(t)
    elif st[key]:
        span[key].append((us(st[key].pop()), us(t)))
cores = sorted({e[0] for e in ev})


def g(c, risc, z, which=0, end=True):
    v = span.get((c, risc, z))
    return v[which][1 if end else 0] if v and len(v) > which else None


metrics = {
    "ready": lambda c: g(c, "BRISC", "writer_r0_ready", end=False),
    "wwait": lambda c: g(c, "TRISC_0", "compute_sd_weight_wait"),
    "slab0": lambda c: g(c, "TRISC_0", "compute_slab_wait") or g(c, "TRISC_0", ""),
    "end": lambda c: g(c, "TRISC_0", "TRISC-KERNEL"),
    "fwd": lambda c: g(c, "NCRISC", "reader_fill"),
}
xs = sorted({c[0] for c in cores})
ys = sorted({c[1] for c in cores})
for name, fn in metrics.items():
    print(f"--- {name}")
    print("     " + " ".join(f"{x:>5}" for x in xs))
    for y in ys:
        cells = []
        for x in xs:
            v = fn((x, y)) if (x, y) in cores else None
            cells.append(f"{v:5.2f}" if v is not None else "    .")
        print(f"{y:>4} " + " ".join(cells))
