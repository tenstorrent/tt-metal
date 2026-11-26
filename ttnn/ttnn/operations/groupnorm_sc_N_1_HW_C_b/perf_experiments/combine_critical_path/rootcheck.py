#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Root vs member cores per run (host-only): p50 of the members' and max of the roots' TRISC_1 c_pass2 / c_pass1 /
c_expand sums, and the kernel end (latest TRISC_1 c_pass2 end) of each class, relative to the run's first zone.

usage: rootcheck.py <profile_log_device.csv> [label,label,...]
"""
import csv
import sys
from collections import defaultdict
from statistics import median


def main(path, labels):
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    by_run = defaultdict(list)
    for r in rows:
        if r["type"] in ("ZONE_START", "ZONE_END"):
            by_run[r["run host ID"]].append(r)
    for i, run in enumerate(sorted(by_run, key=int)):
        st = defaultdict(list)
        dur = defaultdict(float)
        end = {}
        roots = set()
        t0 = None
        for r in by_run[run]:
            if r["RISC processor type"] != "TRISC_1":
                continue
            c = (r["core_x"], r["core_y"])
            key = (r["zone name"], c)
            t = int(r["time[cycles since reset]"])
            t0 = t if t0 is None else min(t0, t)
            if r["zone name"] == "c_finalize":
                roots.add(c)
            if r["type"] == "ZONE_START":
                st[key].append(t)
            elif st[key]:
                dur[key] += (t - st[key].pop()) * k
                end[key] = t
        cores = {c for _, c in dur}
        mem = cores - roots
        out = []
        for z in ("c_pass1", "c_expand", "c_pass2"):
            m = [dur[(z, c)] for c in mem if (z, c) in dur]
            rr = [dur[(z, c)] for c in roots if (z, c) in dur]
            out.append(f"{z[2:]} mem p50 {median(m):.0f} root max {max(rr) if rr else 0:.0f}")
        em = max(end[("c_pass2", c)] for c in mem if ("c_pass2", c) in end)
        er = max((end[("c_pass2", c)] for c in roots if ("c_pass2", c) in end), default=t0)
        lab = labels[i] if i < len(labels) else run
        print(f"{lab:<34s} {' | '.join(out)} | end mem {(em - t0) * k:.0f} root {(er - t0) * k:.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else [])
