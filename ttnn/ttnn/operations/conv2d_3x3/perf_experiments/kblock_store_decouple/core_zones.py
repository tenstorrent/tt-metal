# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core zone timelines of one program run from a device profiler log (no device needed).

    python3 core_zones.py <profile_log_device.csv> <run host ID> <zone>[,<zone>...] [--top N]

Prints, for the N slowest cores (by kernel span) and the median core, each execution of the listed zones as
start-end in us from that core's earliest kernel start."""

import argparse
import collections
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("run", type=int)
    ap.add_argument("zones")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    want = set(args.zones.split(","))
    with open(args.csv) as f:
        header = f.readline()
        freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = csv.reader(f)
        cols = [c.strip() for c in next(rows)]
        ix = {c: i for i, c in enumerate(cols)}
        ev = collections.defaultdict(list)
        for r in rows:
            if len(r) < len(cols) or int(r[ix["run host ID"]]) != args.run:
                continue
            core = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
            ev[core].append(
                (
                    int(r[ix["time[cycles since reset]"]]),
                    r[ix["RISC processor type"]].strip(),
                    r[ix["zone name"]].strip(),
                    r[ix["type"]].strip(),
                )
            )
    t0 = min(t for e in ev.values() for t, *_ in e)
    span = {}
    zones = {}
    for core, e in ev.items():
        e.sort()
        start = min(t for t, risc, z, k in e if z.endswith("-KERNEL") and k == "ZONE_START")
        end = max(t for t, risc, z, k in e if z.endswith("-KERNEL") and k == "ZONE_END")
        span[core] = ((start - t0) / freq, (end - t0) / freq)
        stacks = collections.defaultdict(list)
        out = collections.defaultdict(list)
        for t, risc, z, k in e:
            if z not in want:
                continue
            if k == "ZONE_START":
                stacks[(risc, z)].append(t)
            elif stacks[(risc, z)]:
                s = stacks[(risc, z)].pop()
                out[(risc, z)].append(((s - t0) / freq, (t - t0) / freq))
        zones[core] = out
    order = sorted(span, key=lambda c: span[c][1] - span[c][0])
    pick = order[-args.top :][::-1] + [order[len(order) // 2], order[0]]
    for core in pick:
        print(f"core {core}: kernel {span[core][0]:.1f}-{span[core][1]:.1f} us")
        for (risc, z), iv in sorted(zones[core].items()):
            print(f"   {risc:8s} {z:24s} " + " ".join(f"{a:.0f}-{b:.0f}" for a, b in iv))
    hist = collections.Counter(int((span[c][1] - span[c][0]) // 20) * 20 for c in span)
    print("kernel span histogram (us bucket: cores):", dict(sorted(hist.items())))
    for y in sorted({c[1] for c in span}):
        row = sorted((c[0], round(span[c][1] - span[c][0])) for c in span if c[1] == y)
        print(f"  y={y}: " + " ".join(f"x{x}:{v}" for x, v in row))


if __name__ == "__main__":
    main()
