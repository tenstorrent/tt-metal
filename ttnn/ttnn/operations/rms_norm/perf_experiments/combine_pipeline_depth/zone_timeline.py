# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core ZONE TIMELINE (start/end, not just totals) from a device profiler CSV.

    python3 zone_timeline.py <profile_log_device.csv> [--core X,Y ...] [--row Y]

zone_breakdown.py / zone_percore.py report zone TOTALS.  This idea
(`combine_pipeline_depth`) is about LATENCY OVERLAP, so what it needs is *when*
each zone ran on each core, on one common clock -- specifically:

  * the per-member spread of `writer_gather_ship` completion inside one group
    (does the root's arrival wait measure member SKEW, or the serialized NoC
    ingress at the root?), and
  * the root's compute/writer interleave (is the root's compute IDLE while its
    writer waits for arrivals -- i.e. is there anything to overlap at all?).

All timestamps are cycles-since-reset on the chip's shared clock, rebased to the
earliest KERNEL zone start in the run.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def load(path, run=None):
    """(freq, t0, spans) for ONE launch.  `run` selects the `run host ID`; without
    it the LAST launch in the file is used, and every launch id is printed."""
    lines = open(path).readlines()
    freq = float(lines[0].split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rdr = csv.reader(lines[1:])
    next(rdr)
    rows = [r for r in rdr if len(r) >= 13]
    runs = sorted({r[7].strip() for r in rows}, key=lambda s: int(s))
    if run is None:
        run = runs[-1]
    print(f"launches in file: {runs}   -> using run host ID {run}")
    ev = defaultdict(list)
    for r in rows:
        if r[7].strip() != run:
            continue
        ev[(r[3].strip(), (int(r[1]), int(r[2])), r[10].strip())].append((r[11].strip(), int(r[5])))
    spans = defaultdict(list)
    for k, evs in ev.items():
        s = None
        for typ, c in sorted(evs, key=lambda t: t[1]):
            if typ == "ZONE_START":
                s = c
            elif typ == "ZONE_END" and s is not None:
                spans[k].append((s, c))
                s = None
    t0 = min(v[0][0] for k, v in spans.items() if "KERNEL" in k[2])
    return freq, t0, spans


def main(argv):
    path = argv[0] if argv and argv[0].endswith(".csv") else "generated/profiler/.logs/profile_log_device.csv"
    cores, rows = [], []
    run = None
    i = 0
    while i < len(argv):
        if argv[i] == "--run":
            run = argv[i + 1]
            i += 2
        elif argv[i] == "--core":
            x, y = argv[i + 1].split(",")
            cores.append((int(x), int(y)))
            i += 2
        elif argv[i] == "--row":
            rows.append(int(argv[i + 1]))
            i += 2
        else:
            i += 1

    freq, t0, spans = load(path, run)

    def ns(c):
        return (c - t0) / freq * 1000

    ks = sorted(ns(v[0][0]) for k, v in spans.items() if k[2].endswith("-KERNEL"))
    print(f"KERNEL start spread across cores: {ks[0]:.0f} .. {ks[-1]:.0f} ns (n={len(ks)})")
    roots = sorted({k[1] for k in spans if "root_sum" in k[2] or "root_fold" in k[2]})
    print(f"root/gatherer cores: {roots}")

    for y in rows:
        root = next((c for c in roots if c[1] == y), None)
        print(f"\n=== group on virtual row y={y} (root {root}) ===")
        ships = sorted((k[1], v) for k, v in spans.items() if k[2] == "writer_gather_ship" and k[1][1] == y)
        gw = next((v for k, v in spans.items() if k[2] == "writer_gather_wait" and k[1] == root), [])
        nb = max((len(v) for _, v in ships), default=0)
        for blk in range(nb):
            ends = sorted((ns(v[blk][1]), c) for c, v in ships if blk < len(v))
            if not ends:
                continue
            w = (
                f"{ns(gw[blk][0]):.0f}->{ns(gw[blk][1]):.0f} ({(gw[blk][1]-gw[blk][0])/freq*1000:.0f} ns)"
                if blk < len(gw)
                else "-"
            )
            print(
                f" blk {blk}: ship_END {ends[0][0]:.0f}..{ends[-1][0]:.0f}  spread {ends[-1][0]-ends[0][0]:.0f} ns | root gather_wait {w}"
            )
            print("     " + " ".join(f"x{c[0]}:{t:.0f}" for t, c in ends))

    for core in cores:
        tl = sorted(
            (ns(s), ns(e), k[0], k[2], i)
            for k, v in spans.items()
            if k[1] == core and "-FW" not in k[2]
            for i, (s, e) in enumerate(v)
        )
        print(f"\n===== core {core} timeline =====")
        for s, e, risc, z, i in tl:
            print(f"  {s:8.0f} -> {e:8.0f} ({e-s:7.0f})  {risc:7s} {z}#{i}")


if __name__ == "__main__":
    main(sys.argv[1:])
