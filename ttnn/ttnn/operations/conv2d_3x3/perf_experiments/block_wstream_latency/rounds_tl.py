# usage: rounds_tl.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [zone,...]
# Per core (logical x, y from the sorted physical coordinates of the run): the writer's weight-fetch zone ends (round
# landed and pushed), its spread mcast / relay / issue zones, and the kernel end, in us after the run's first kernel
# start. Host-only.
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
xs = sorted({e[0][0] for e in ev})
ys = sorted({e[0][1] for e in ev})
lx = {x: i for i, x in enumerate(xs)}
ly = {y: i for i, y in enumerate(ys)}
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
stack = collections.defaultdict(list)
zones = collections.defaultdict(list)
kend = collections.defaultdict(float)
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_END":
        kend[core] = max(kend[core], us(t))
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        zones[core].append((risc, z, us(s), us(t)))
want = sys.argv[3].split(",") if len(sys.argv) > 3 else ["writer_weight_fetch"]
print(f"run {big[idx-1]}  x {xs}  y {ys}")
for core in sorted(zones, key=lambda c: (lx[c[0]], ly[c[1]])):
    parts = []
    for w in want:
        zs = sorted([z for z in zones[core] if z[1] == w], key=lambda z: z[2])
        if zs:
            parts.append(w.replace("writer_", "w.") + " " + " ".join(f"{a:.2f}-{b:.2f}" for _, _, a, b in zs))
    print(f"({lx[core[0]]:2d},{ly[core[1]]}) phys{core} end {kend[core]:.2f}  " + " | ".join(parts))
