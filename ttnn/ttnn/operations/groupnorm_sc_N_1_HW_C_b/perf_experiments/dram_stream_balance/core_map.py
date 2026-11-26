#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core timeline of one profiled dispatch on a grid map (host-only; reads a profile_log_device.csv).

usage: core_map.py <report dir or profile_log_device.csv> [--run <run host ID>] [--zone NAME:RISC ...]

Prints, per physical (x, y), the end time (us, from the earliest kernel start) of selected zones, plus the
op's DEVICE KERNEL DURATION from the ops CSV next to the log. Default maps:
  p1_end  = end of the writer's w_wait_partials (this core's pass 1 + collapse done)
  stats   = end of w_wait_gather / w_mcast_recv (unit stats available)
  out     = w_output duration (pass-2 writes, incl. the wait for compute)
  end     = BRISC kernel end
"""

import csv
import glob
import os
import sys
from collections import defaultdict


def load(path, run=None):
    if os.path.isdir(path):
        path = os.path.join(path, "profile_log_device.csv")
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    runs = sorted({r["run host ID"] for r in rows}, key=int)
    run = run or runs[-1]
    rows = [r for r in rows if r["run host ID"] == run]
    k = 1000.0 / mhz
    z = defaultdict(list)  # (core, risc, name) -> [(start, end)]
    st = defaultdict(list)
    for r in rows:
        if r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        key = ((int(r["core_x"]), int(r["core_y"])), r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            st[key].append(t)
        elif st[key]:
            z[key].append((st[key].pop(), t))
    t0 = min(s for (c, risc, n), v in z.items() if n.endswith("-KERNEL") for s, _ in v)
    return z, t0, k, os.path.dirname(path)


def grid_print(title, vals):
    xs = sorted({c[0] for c in vals})
    ys = sorted({c[1] for c in vals})
    print(f"\n== {title} (us)   x: " + " ".join(f"{x:5d}" for x in xs))
    for y in ys:
        print(f"   y={y:2d}            " + " ".join(f"{vals[(x, y)]:5.0f}" if (x, y) in vals else "    ." for x in xs))
    v = list(vals.values())
    if v:
        print(f"   min {min(v):.0f}  max {max(v):.0f}  mean {sum(v) / len(v):.0f}")


def main(argv):
    path = argv[0]
    run = argv[argv.index("--run") + 1] if "--run" in argv else None
    z, t0, k, d = load(path, run)
    for csvp in glob.glob(os.path.join(d, "ops_perf_results*.csv")):
        for r in csv.DictReader(open(csvp)):
            print("DEVICE KERNEL DURATION [ns]:", r["DEVICE KERNEL DURATION [ns]"])

    def end_of(risc, name):
        return {c: (max(e for _, e in v) - t0) * k / 1000 for (c, r, n), v in z.items() if r == risc and n == name}

    def dur_of(risc, name):
        return {c: sum(e - s for s, e in v) * k / 1000 for (c, r, n), v in z.items() if r == risc and n == name}

    grid_print("pass-1 end (w_wait_partials end)", end_of("BRISC", "w_wait_partials"))
    stats = end_of("BRISC", "w_mcast_recv")
    stats.update(end_of("BRISC", "w_mcast_send"))
    grid_print("stats in hand (mcast end)", stats)
    grid_print("w_output duration", dur_of("BRISC", "w_output"))
    grid_print("BRISC kernel end", end_of("BRISC", "BRISC-KERNEL"))
    grid_print("NCRISC kernel end", end_of("NCRISC", "NCRISC-KERNEL"))
    for extra in [a for a in argv[1:] if ":" in a]:
        name, risc = extra.split(":")
        grid_print(f"{name} duration on {risc}", dur_of(risc, name))


if __name__ == "__main__":
    main(sys.argv[1:])
