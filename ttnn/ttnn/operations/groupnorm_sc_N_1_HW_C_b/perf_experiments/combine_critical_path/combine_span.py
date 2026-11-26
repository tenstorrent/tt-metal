#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Combine-path span per run (host-only): on every core's math thread (TRISC_1), the time from the end of c_pass1 to
the start of c_pass2 (collapse + gather/finalize/mcast round trip + expand), and the first-to-last c_pass2 start
across cores. p50 / max over cores, ns. Independent of the pass-1 / pass-2 loop rates, which move with code layout.

usage: combine_span.py <profile_log_device.csv> [label,label,...]
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
        if r["type"] in ("ZONE_START", "ZONE_END") and r["RISC processor type"] == "TRISC_1":
            by_run[r["run host ID"]].append(r)
    for i, run in enumerate(sorted(by_run, key=int)):
        p1_end, p2_start, ex = {}, {}, {}
        for r in by_run[run]:
            c = (r["core_x"], r["core_y"])
            t = int(r["time[cycles since reset]"])
            if r["zone name"] == "c_pass1" and r["type"] == "ZONE_END":
                p1_end[c] = t
            elif r["zone name"] == "c_pass2" and r["type"] == "ZONE_START":
                p2_start.setdefault(c, t)
        span = [(p2_start[c] - p1_end[c]) * k for c in p1_end if c in p2_start]
        lab = labels[i] if i < len(labels) else run
        print(f"{lab:<36s} p1end->p2start p50 {median(span):6.0f} max {max(span):6.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else [])
