#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone breakdown of one profiled dispatch (host-only; reads profile_log_device.csv).

usage: zone_breakdown.py [profile_log_device.csv] [--run <run host ID>] [--ops-csv <ops_perf_results.csv>]

For every (zone, RISC) it prints: cores that ran it, executions per core, and the per-core SUM of its durations
(p50 and max across cores) in ns. Per RISC it prints the *-KERNEL span max / p50 (core balance) and the marker
count per (core, RISC) with the last user zone's end relative to the kernel span (marker-budget exhaustion is
silent - a zone set that stops short of the span is partial). Zone numbers are OCCUPANCY (wait + work).
"""

import csv
import sys
from collections import defaultdict
from statistics import median


def main(argv):
    path = "generated/profiler/.logs/profile_log_device.csv"
    run = None
    args = list(argv)
    if args and not args[0].startswith("--"):
        path = args.pop(0)
    if "--run" in args:
        run = args[args.index("--run") + 1]
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) if "CHIP_FREQ" in head else 1350.0
        rows = list(csv.DictReader(f, skipinitialspace=True))
    runs = sorted({r["run host ID"] for r in rows}, key=lambda x: int(x) if x.isdigit() else 0)
    if run is None:
        run = runs[-1]
    rows = [r for r in rows if r["run host ID"] == run]
    cyc2ns = 1000.0 / mhz

    # Pair zone starts/ends per (core, risc, zone) with a stack.
    stacks = defaultdict(list)
    dur = defaultdict(lambda: defaultdict(float))  # (zone, risc) -> core -> ns
    cnt = defaultdict(lambda: defaultdict(int))
    span = {}  # (core, risc) -> (start, end) of the *-KERNEL zone
    last_user_end = defaultdict(int)
    markers = defaultdict(int)
    for r in rows:
        core = (int(r["core_x"]), int(r["core_y"]))
        risc = r["RISC processor type"]
        name = r["zone name"]
        typ = r["type"]
        t = int(r["time[cycles since reset]"])
        if typ not in ("ZONE_START", "ZONE_END"):
            continue
        key = (core, risc, name)
        if name.endswith("-KERNEL"):
            if typ == "ZONE_START":
                span[(core, risc)] = [t, None]
            else:
                span[(core, risc)][1] = t
            continue
        if name.endswith("-FW"):
            continue
        markers[(core, risc)] += 1
        if typ == "ZONE_START":
            stacks[key].append(t)
        else:
            if stacks[key]:
                s = stacks[key].pop()
                dur[(name, risc)][core] += (t - s) * cyc2ns
                cnt[(name, risc)][core] += 1
                last_user_end[(core, risc)] = max(last_user_end[(core, risc)], t)

    print(f"run host ID {run} (of {runs}), {mhz:.0f} MHz")
    print("\n== kernel span per RISC (ns): p50 / max / max:p50, cores")
    by_risc = defaultdict(list)
    for (core, risc), (s, e) in span.items():
        if e is not None:
            by_risc[risc].append((e - s) * cyc2ns)
    for risc in sorted(by_risc):
        v = by_risc[risc]
        print(f"  {risc:7s} p50 {median(v):10.0f}  max {max(v):10.0f}  {max(v) / max(median(v), 1):5.2f}  n={len(v)}")

    print("\n== zones (per-core SUM of durations, ns): cores, exec/core, p50, max")
    order = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
    for risc in order:
        keys = [k for k in dur if k[1] == risc]
        if not keys:
            continue
        print(f"  -- {risc}")
        first = {}
        for k in keys:
            first[k] = min(dur[k].keys())
        for k in sorted(keys, key=lambda k: -max(dur[k].values())):
            v = list(dur[k].values())
            ex = median(list(cnt[k].values()))
            print(f"    {k[0]:22s} cores {len(v):4d}  exec {ex:5.1f}  p50 {median(v):10.0f}  max {max(v):10.0f}")

    print("\n== marker budget: max user markers per (core, RISC); last user zone end vs kernel span end")
    worst = defaultdict(int)
    short = defaultdict(float)
    for (core, risc), m in markers.items():
        worst[risc] = max(worst[risc], m)
        sp = span.get((core, risc))
        if sp and sp[1] is not None and sp[1] > sp[0]:
            frac = (last_user_end[(core, risc)] - sp[0]) / (sp[1] - sp[0])
            short[risc] = max(short[risc], 1 - frac)
    for risc in order:
        if risc in worst:
            print(
                f"  {risc:7s} max markers {worst[risc]:4d} (cap 250)  worst gap after last zone {short[risc] * 100:5.1f}% of span"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
