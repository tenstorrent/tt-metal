# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per-core zone intervals (us after the run's earliest kernel start) of one run of a device profile CSV.
    python3 zq.py <profile_log_device.csv[.gz]> <run_index (1-based, runs with >= 8 cores)> <zone substr,...> [risc]"""
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
zs = sys.argv[3].split(",")
want_risc = sys.argv[4] if len(sys.argv) > 4 else None
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
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
stack = collections.defaultdict(list)
out = collections.defaultdict(list)
kend = {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_END":
        kend[core] = max(kend.get(core, 0), us(t))
    if not any(s == z or (s.endswith("*") and z.startswith(s[:-1])) for s in zs):
        continue
    if want_risc and risc != want_risc:
        continue
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        out[core].append((us(s), us(t), z))
for core in sorted(out, key=lambda c: (c[0], c[1])):
    items = sorted(out[core])
    print(
        f"{core} end {kend.get(core, 0):5.2f}: "
        + " ".join(f"{z.split('_',1)[1][:12]}[{a:.2f}-{b:.2f}]" for a, b, z in items)
    )
