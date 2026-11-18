# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-occurrence zone list for one core of one run (no ttnn).

    python3 zone_seq.py <profile_log_device.csv> <run index 1-based among runs with >= 8 cores> [x,y] [zone,zone,...]

Prints every occurrence of the selected zones (default: tilize / gather zones) on that core (default: the core whose
kernel ends last), per RISC, as start-end us relative to the run's earliest kernel start.
"""
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
want = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
zones = (
    set(sys.argv[4].split(","))
    if len(sys.argv) > 4
    else {
        "compute_tilize",
        "compute_tilize_wait",
        "reader_gather",
        "reader_gather_reserve",
        "reader_gather_rwait",
        "compute_untilize",
        "reader_center",
        "reader_center_wait",
        "compute_matmul_walk",
    }
)
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
kst, ken = {}, {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL"):
        if kind == "ZONE_START":
            kst[(core, risc)] = min(kst.get((core, risc), t), t)
        else:
            ken[(core, risc)] = max(ken.get((core, risc), t), t)
t0 = min(kst.values())
us = lambda c: (c - t0) / freq
if want is None:
    want = max(ken, key=lambda k: ken[k])[0]
print(f"run {big[idx-1]} core {want}")
stack = collections.defaultdict(list)
out = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if core != want or z not in zones:
        continue
    if kind == "ZONE_START":
        stack[(risc, z)].append(t)
    elif stack[(risc, z)]:
        s = stack[(risc, z)].pop()
        out[(risc, z)].append((us(s), us(t)))
for (risc, z), occ in sorted(out.items()):
    tot = sum(b - a for a, b in occ)
    print(
        f"  {risc:7} {z:24} n={len(occ):3} sum={tot:6.2f} us : " + " ".join(f"{a:.2f}+{(b-a)*1000:.0f}" for a, b in occ)
    )
