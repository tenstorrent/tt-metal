#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Combine-path timeline of one root core per run (host-only): every zone execution on every RISC of the first
core that ran c_finalize, as start/end ns relative to that core's earliest zone start.

usage: timeline.py <profile_log_device.csv> <run_index> [zone_substr,...]
"""
import csv
import sys
from collections import defaultdict


def main(path, idx, filt):
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    runs = sorted({r["run host ID"] for r in rows}, key=int)
    run = runs[idx]
    rows = [r for r in rows if r["run host ID"] == run and r["type"] in ("ZONE_START", "ZONE_END")]
    roots = sorted({(r["core_x"], r["core_y"]) for r in rows if r["zone name"] == "c_finalize"})
    core = roots[0]
    rows = [r for r in rows if (r["core_x"], r["core_y"]) == core]
    t0 = min(int(r["time[cycles since reset]"]) for r in rows)
    stacks = defaultdict(list)
    ev = []
    for r in rows:
        key = (r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            stacks[key].append(t)
        elif stacks[key]:
            s = stacks[key].pop()
            ev.append((s, t, key))
    ev.sort()
    print(f"run {run} root core {core} ({len(roots)} roots)")
    for s, t, (risc, name) in ev:
        if filt and not any(z in name for z in filt):
            continue
        print(f"{risc:>8s} {name:<22s} {(s - t0) * k:9.0f} {(t - t0) * k:9.0f}  dur {(t - s) * k:7.0f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else [])
