#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-run zone timeline: p50 over cores of (start, end) relative to the run's earliest kernel start, p50 duration,
and max end, split root / member (root = the core has a TRISC_1 c_finalize zone, or every core when cpi == 1).

usage: zones.py <profile_log_device.csv> [labels,comma,separated] [--zones z1,z2,...]
"""
import csv
import sys
from collections import defaultdict
from statistics import median

DEFAULT = [
    ("NCRISC", "r_block"),
    ("NCRISC", "r_pilot_bar"),
    ("NCRISC", "r_shift_rows"),
    ("NCRISC", "r_map_setup"),
    ("NCRISC", "r_ones_shift_affine"),
    ("NCRISC", "r_ones"),
    ("NCRISC", "r_gshift"),
    ("NCRISC", "r_zero_all"),
    ("NCRISC", "r_aff_zero"),
    ("NCRISC", "r_aff_rd"),
    ("NCRISC", "r_aff_build"),
    ("TRISC_0", "c_wait_pilot"),
    ("TRISC_0", "c_wait_map"),
    ("TRISC_0", "c_wait_stats"),
    ("TRISC_0", "c_wait_affine"),
    ("TRISC_1", "c_pass1"),
    ("TRISC_1", "c_expand"),
    ("TRISC_1", "c_pass2"),
]


def main(argv):
    path = argv[0]
    labels = argv[1].split(",") if len(argv) > 1 and not argv[1].startswith("--") else []
    cols = DEFAULT
    if "--zones" in argv:
        want = argv[argv.index("--zones") + 1].split(",")
        cols = [c for c in DEFAULT if c[1] in want]
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    t0 = {}
    kend = defaultdict(int)
    for r in rows:
        if r["type"] in ("ZONE_START", "ZONE_END") and r["zone name"].endswith("-KERNEL"):
            run, t = r["run host ID"], int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                t0[run] = min(t0.get(run, t), t)
            else:
                kend[run] = max(kend[run], t)
    st = defaultdict(list)
    ev = defaultdict(lambda: defaultdict(list))  # (run, risc, zone) -> core -> [(s, e)]
    roots = defaultdict(set)
    for r in rows:
        if r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        run, risc, n = r["run host ID"], r["RISC processor type"], r["zone name"]
        core = (int(r["core_x"]), int(r["core_y"]))
        t = int(r["time[cycles since reset]"])
        key = (run, risc, n, core)
        if r["type"] == "ZONE_START":
            st[key].append(t)
        elif st[key]:
            s = st[key].pop()
            ev[(run, risc, n)][core].append(((s - t0[run]) * k, (t - t0[run]) * k))
            if risc == "TRISC_1" and n == "c_finalize":
                roots[run].add(core)
    runs = sorted(t0, key=int)
    for i, run in enumerate(runs):
        lab = labels[i] if i < len(labels) else run
        span = (kend[run] - t0[run]) * k
        print(f"=== {lab}  (run {run}, span {span:,.0f} ns)")
        print(
            f"  {'zone':22s} {'role':6s} {'n':>3s} {'start50':>8s} {'end50':>8s} {'dur50':>7s} {'durmax':>7s} {'endmax':>8s}"
        )
        for risc, n in cols:
            d = ev.get((run, risc, n))
            if not d:
                continue
            for role in ("root", "member"):
                cores = [c for c in d if (c in roots[run]) == (role == "root")]
                if not cores:
                    continue
                # first execution per core (block 0), duration summed over executions
                ss = [d[c][0][0] for c in cores]
                ee = [d[c][0][1] for c in cores]
                du = [sum(e - s for s, e in d[c]) for c in cores]
                print(
                    f"  {n:22s} {role:6s} {len(cores):3d} {median(ss):8.0f} {median(ee):8.0f} {median(du):7.0f}"
                    f" {max(du):7.0f} {max(ee):8.0f}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
