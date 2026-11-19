# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per grid row (core_y) critical end time and per-row zone sums from a device profile log (no ttnn).

    python3 row_report.py <profile_log_device.csv[.gz]> [--zones compute_tilize,compute_matmul_walk]

For every run on >= 50 cores: per core_y, the latest kernel end (us after the run's first kernel start) over the
row's cores, and the max over the row's cores of each listed zone's per-core sum (TRISC_0).
"""

import argparse
import collections
import csv
import gzip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--zones", default="compute_untilize,compute_tilize,compute_tilize_wait,compute_matmul_walk")
    args = ap.parse_args()
    opener = gzip.open if args.log.endswith(".gz") else open
    with opener(args.log, "rt") as f:
        header = f.readline()
        freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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
    zones = args.zones.split(",")
    for run_id in sorted(runs):
        ev = sorted(runs[run_id], key=lambda e: e[2])
        cores = {e[0] for e in ev}
        if len(cores) < 50:
            continue
        t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
        end = collections.defaultdict(int)
        stacks = collections.defaultdict(list)
        sums = collections.defaultdict(float)
        for core, risc, t, zone, kind in ev:
            if zone.endswith("-KERNEL") and kind == "ZONE_END":
                end[core] = max(end[core], t)
            if risc == "TRISC_0" and zone in zones:
                if kind == "ZONE_START":
                    stacks[(core, zone)].append(t)
                elif kind == "ZONE_END" and stacks[(core, zone)]:
                    sums[(core, zone)] += t - stacks[(core, zone)].pop()
        by_y = collections.defaultdict(list)
        for c in cores:
            by_y[c[1]].append(c)
        print(f"=== run {run_id}: {len(cores)} cores; per core_y: max kernel end (us) | max zone sums (us, TRISC_0)")
        print(f"  {'y':>3} {'end':>7} " + " ".join(f"{z[-14:]:>14}" for z in zones))
        for y in sorted(by_y):
            cs = by_y[y]
            e = max(end[c] - t0 for c in cs) / freq
            zs = [max(sums[(c, z)] for c in cs) / freq for z in zones]
            print(f"  {y:>3} {e:7.2f} " + " ".join(f"{v:14.2f}" for v in zs))


if __name__ == "__main__":
    main()
