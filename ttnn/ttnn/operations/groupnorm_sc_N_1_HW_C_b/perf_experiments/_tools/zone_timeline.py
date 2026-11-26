#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-run zone timeline: first start / last end of each (RISC, zone), ns relative to the earliest kernel start of the
run, p50 over cores (host-only; reads profile_log_device.csv).

usage: zone_timeline.py <profile_log_device.csv> [label,label,...]
"""
import csv
import sys
from collections import defaultdict
from statistics import median

path = sys.argv[1]
labels = sys.argv[2].split(",") if len(sys.argv) > 2 else None
with open(path) as f:
    head = f.readline()
    mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
hdr = [h.strip() for h in rows[0]]
ix = {h: i for i, h in enumerate(hdr)}
runs = defaultdict(list)
for r in rows[1:]:
    runs[r[ix["run host ID"]].strip()].append(r)
for k, rid in enumerate(sorted(runs, key=int)):
    ev = runs[rid]
    # (core, risc, zone) -> [first start, last end]
    span = {}
    open_ = {}
    t0 = None
    for r in ev:
        core = (r[ix["core_x"]].strip(), r[ix["core_y"]].strip())
        risc = r[ix["RISC processor type"]].strip()
        z = r[ix["zone name"]].strip()
        t = int(r[ix["time[cycles since reset]"]])
        typ = r[ix["type"]].strip()
        if z.endswith("-KERNEL") and typ == "ZONE_START":
            t0 = t if t0 is None else min(t0, t)
        key = (core, risc, z)
        if typ == "ZONE_START":
            s = span.setdefault(key, [t, t])
            s[0] = min(s[0], t)
        elif typ == "ZONE_END":
            s = span.setdefault(key, [t, t])
            s[1] = max(s[1], t)
    agg = defaultdict(lambda: ([], []))
    for (core, risc, z), (a, b) in span.items():
        if z.endswith("-FW"):
            continue
        agg[(risc, z)][0].append((a - t0) / mhz * 1000)
        agg[(risc, z)][1].append((b - t0) / mhz * 1000)
    name = labels[k] if labels and k < len(labels) else rid
    print(f"== {name}")
    for (risc, z), (a, b) in sorted(agg.items(), key=lambda kv: median(kv[1][0])):
        print(f"   {risc:8s} {z:28s} start {median(a):9.0f}  end {median(b):9.0f}")
