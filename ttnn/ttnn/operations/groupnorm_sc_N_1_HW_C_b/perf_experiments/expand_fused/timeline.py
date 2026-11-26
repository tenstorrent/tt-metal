#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zone timeline of one dispatch on one or more cores (host-only).

usage: timeline.py <report_dir> <dispatch_index (0-based)> [core_x,core_y ...]   (default: median-span core)
Prints every zone (start / end ns relative to the dispatch's earliest kernel start) per RISC.
"""

import csv
import os
import sys
from collections import defaultdict


def main(report, idx, cores):
    with open(os.path.join(report, "profile_log_device.csv")) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        dev = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    runs = sorted({r["run host ID"] for r in dev}, key=int)
    run = runs[idx]
    rows = [r for r in dev if r["run host ID"] == run and r["type"] in ("ZONE_START", "ZONE_END")]
    t0 = min(int(r["time[cycles since reset]"]) for r in rows if r["zone name"].endswith("-KERNEL"))
    stacks = defaultdict(list)
    ev = defaultdict(list)
    for r in rows:
        core = f'{r["core_x"]},{r["core_y"]}'
        if cores and core not in cores:
            continue
        name = r["zone name"]
        if name.endswith("-FW"):
            continue
        key = (core, r["RISC processor type"], name)
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            stacks[key].append(t)
        elif stacks[key]:
            s = stacks[key].pop()
            ev[core].append(((s - t0) * k, (t - t0) * k, r["RISC processor type"], name))
    for core in sorted(ev):
        print(f"== core {core}")
        for s, e, risc, name in sorted(ev[core]):
            print(f"  {risc:8s} {name:24s} {s:9.0f} -> {e:9.0f}  ({e - s:7.0f})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
