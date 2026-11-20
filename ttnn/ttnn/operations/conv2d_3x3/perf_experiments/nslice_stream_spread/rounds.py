# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per run of a device profile, each weight round's landing (end of the r-th writer_weight_fetch zone, us
after the run's earliest kernel start) as p50 / max over grid and spare-column cores (spare = x past the column
stream's grid: cores with no ring...: given by --spare-x), plus the walk end and the kernel end.
    python3 rounds.py <profile_log_device.csv[.gz]> [run_index ...] [--spare-x 13,14,15]"""
import collections, csv, gzip, statistics, sys

args = [a for a in sys.argv[1:]]
spare_x = {13, 14, 15}
if "--spare-x" in args:
    i = args.index("--spare-x")
    spare_x = {int(v) for v in args[i + 1].split(",")}
    del args[i : i + 2]
path = args[0]
want = [int(v) for v in args[1:]]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
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
for n, k in enumerate(big, 1):
    if want and n not in want:
        continue
    ev = sorted(runs[k], key=lambda e: e[2])
    t0 = min(t for c, r, t, z, kd in ev if z.endswith("-KERNEL") and kd == "ZONE_START")
    us = lambda c: (c - t0) / freq
    land = collections.defaultdict(list)
    walk_end, kend = {}, {}
    st = {}
    for core, risc, t, z, kd in ev:
        if z.endswith("-KERNEL") and kd == "ZONE_END":
            kend[core] = max(kend.get(core, 0), us(t))
        if z == "writer_weight_fetch" and risc == "BRISC":
            if kd == "ZONE_START":
                st[core] = t
            else:
                land[core].append(us(t))
        if z == "compute_matmul_walk" and kd == "ZONE_END":
            walk_end[core] = max(walk_end.get(core, 0), us(t))
    D = max(kend.values())
    print(
        f"run {n}: kernel end max {D:.2f}  walk end p50 {statistics.median(walk_end.values()):.2f} max {max(walk_end.values()):.2f}"
    )
    for grp, sel in (("grid ", lambda c: c[0] not in spare_x), ("spare", lambda c: c[0] in spare_x)):
        cs = [c for c in land if sel(c)]
        if not cs:
            continue
        nr = max(len(land[c]) for c in cs)
        line = []
        for r in range(nr):
            v = [land[c][r] for c in cs if len(land[c]) > r]
            line.append(f"r{r} {statistics.median(v):.2f}/{max(v):.2f}")
        crit = max(cs, key=lambda c: kend.get(c, 0))
        print(f"  {grp} " + "  ".join(line) + f"   | crit {crit} end {kend[crit]:.2f}")
