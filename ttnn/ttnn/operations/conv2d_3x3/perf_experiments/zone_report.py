# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage zone report from the device profiler log (no device needed).

    python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/zone_report.py \
        [generated/profiler/.logs/profile_log_device.csv] [--min-cores N] [--run RUN_ID]

For every run (one program dispatch) on at least --min-cores cores it prints:
  - each RISC's *-KERNEL span: max / p50 over cores (max/p50 > ~1.3 = imbalanced work);
  - each zone, per RISC: per-core SUM of its executions (us), p50 / max over cores, executions per core;
  - the busiest core's marker count, so a silently truncated profile (cap 250 per RISC) is visible.
A zone's number is occupancy (it includes any wait inside its braces), not payload.
"""

import argparse
import collections
import csv
import statistics
import sys


def load(path):
    with open(path) as f:
        header = f.readline()
        freq_mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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
    return freq_mhz, runs


def summarize(events, freq_mhz):
    to_us = 1.0 / freq_mhz
    stacks = collections.defaultdict(list)
    sums = collections.defaultdict(float)  # (core, risc, zone) -> cycles
    counts = collections.Counter()
    markers = collections.Counter()
    for core, risc, t, zone, kind in sorted(events, key=lambda e: e[2]):
        markers[(core, risc)] += 1
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stacks[key].append(t)
        elif kind == "ZONE_END" and stacks[key]:
            sums[key] += t - stacks[key].pop()
            counts[key] += 1
    per_zone = collections.defaultdict(list)
    execs = collections.defaultdict(list)
    for (core, risc, zone), cyc in sums.items():
        per_zone[(risc, zone)].append(cyc * to_us)
        execs[(risc, zone)].append(counts[(core, risc, zone)])
    return per_zone, execs, markers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="generated/profiler/.logs/profile_log_device.csv")
    ap.add_argument("--min-cores", type=int, default=8)
    ap.add_argument("--run", type=int, default=None)
    args = ap.parse_args()
    freq_mhz, runs = load(args.log)
    for run_id in sorted(runs):
        if args.run is not None and run_id != args.run:
            continue
        per_zone, execs, markers = summarize(runs[run_id], freq_mhz)
        cores = {c for (c, _) in markers}
        if len(cores) < args.min_cores:
            continue
        print(f"=== run {run_id}: {len(cores)} cores, {freq_mhz:.0f} MHz")
        kernel_rows = sorted(k for k in per_zone if k[1].endswith("-KERNEL"))
        for risc, zone in kernel_rows:
            v = per_zone[(risc, zone)]
            print(f"  {zone:<32} max {max(v):8.2f} us  p50 {statistics.median(v):8.2f} us  cores {len(v)}")
        print(f"  {'zone':<34}{'risc':<8}{'p50 us':>9}{'max us':>9}{'min us':>9}{'exec/core':>10}{'cores':>6}")
        for risc, zone in sorted(per_zone, key=lambda k: (k[0], k[1])):
            if zone.endswith("-KERNEL") or zone.endswith("-FW"):
                continue
            v = per_zone[(risc, zone)]
            e = statistics.median(execs[(risc, zone)])
            print(f"  {zone:<34}{risc:<8}{statistics.median(v):9.2f}{max(v):9.2f}{min(v):9.2f}{e:10.0f}{len(v):6d}")
        top = markers.most_common(1)[0]
        print(f"  busiest marker count: {top[1]} on {top[0]} (cap ~250 user markers per RISC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
