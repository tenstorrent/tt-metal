# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core grid maps from generated/profiler/.logs/profile_log_device.csv (host-side only).

    python3 core_map.py [--run N] [--zone NAME --risc RISC] ...
Prints, per run: a grid (physical x columns, y rows) of the BRISC-KERNEL end (us from the program
start), and of the writer's claimed sticks (TS_DATA writer_split_sticks) when present; plus any
extra (zone, risc) summed-duration maps given with --zone/--risc pairs.
"""
import csv
import sys
from collections import defaultdict

args = sys.argv[1:]
path = "generated/profiler/.logs/profile_log_device.csv"
only_run, extra = None, []
i = 0
while i < len(args):
    if args[i] == "--run":
        only_run = int(args[i + 1])
        i += 2
    elif args[i] == "--zone":
        extra.append((args[i + 1], args[i + 3]))
        i += 4
    else:
        path = args[i]
        i += 1
f = open(path)
header = f.readline()
mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) if "CHIP_FREQ" in header else 1350.0
rows = list(csv.DictReader(f, skipinitialspace=True))
runs = sorted({int(r["run host ID"]) for r in rows}) if only_run is None else [only_run]
for run in runs:
    ev = defaultdict(list)
    data = {}
    for r in rows:
        if int(r["run host ID"]) != run:
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
        if r["type"] == "TS_DATA":
            data[c] = int(r["data"])
            continue
        ev[(c, r["RISC processor type"], r["zone name"])].append((r["type"], int(r["time[cycles since reset]"])))
    t0 = min(t for k, v in ev.items() if k[2].endswith("KERNEL") for _, t in v)
    end, tot = {}, defaultdict(int)
    for k, v in ev.items():
        v.sort(key=lambda x: x[1])
        st = []
        for typ, t in v:
            if typ == "ZONE_START":
                st.append(t)
            elif st:
                s = st.pop()
                tot[k] += t - s
                if k[2] == "BRISC-KERNEL":
                    end[k[0]] = (t - t0) / mhz
    xs = sorted({c[0] for c in end})
    ys = sorted({c[1] for c in end})

    def grid(title, val, fmt="{:6.1f}"):
        print(f"--- run {run}: {title}")
        print("   y\\x " + "".join(f"{x:7d}" for x in xs))
        for y in ys:
            print(f"  {y:4d} " + "".join((" " + fmt.format(val[(x, y)])) if (x, y) in val else "      -" for x in xs))

    grid("BRISC-KERNEL end (us)", end)
    if data:
        grid("writer-claimed sticks", data, "{:6d}")
    for z, risc in extra:
        grid(f"{risc} {z} sum (us)", {k[0]: v / mhz for k, v in tot.items() if k[1] == risc and k[2] == z})
