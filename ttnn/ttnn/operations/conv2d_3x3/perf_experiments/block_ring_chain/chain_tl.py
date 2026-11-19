# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring-chain timeline from a device profiler log (host only).

    python3 chain_tl.py <profile_log_device.csv> <run_index 1-based among runs with >= 8 cores> [x,y ...] [--zones a,b]
                        [--cores] [--steps]

--cores: every core's kernel end (us from the run's earliest start), sorted.
--steps: per core given, the start times of every zone listed in --zones (default the ring zones), full precision.
"""
import collections
import csv
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
path, idx = args[0], int(args[1])
want = [tuple(int(v) for v in a.split(",")) for a in args[2:]]
zones = None
for f in flags:
    if f.startswith("--zones="):
        zones = f.split("=", 1)[1].split(",")
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
print(f"run {big[idx-1]} ({len({e[0] for e in ev})} cores)")
if "--cores" in flags:
    cend = collections.defaultdict(float)
    for (core, risc), t in ken.items():
        cend[core] = max(cend[core], us(t))
    for core, t in sorted(cend.items(), key=lambda kv: kv[1]):
        print(f"  {core}: end {t:.2f}")
stack = collections.defaultdict(list)
spans = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if want and core not in want:
        continue
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        spans[(core, risc, z)].append((us(s), us(t)))
for core in want:
    print(f"core {core}")
    for (c, risc, z), sp in sorted(spans.items()):
        if c != core or z.endswith("-FW") or z.endswith("-KERNEL"):
            continue
        if zones and not any(q in z for q in zones):
            continue
        print(f"  {risc:7s} {z:28s} " + " ".join(f"{a:.2f}-{b:.2f}" for a, b in sp))
