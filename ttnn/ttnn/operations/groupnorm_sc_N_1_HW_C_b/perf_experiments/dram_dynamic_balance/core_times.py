#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core zone durations (us) of one run, keyed by LOGICAL core "x,y" (physical x / y ranks: the profiler reports
physical NoC0 coordinates; logical x / y are their sorted ranks on this unharvested-rectangle grid).

usage: core_times.py <profile_log_device.csv> <run host ID> [--json out.json]
prints per-core TRISC_1 c_pass1 / c_pass2 and TRISC_0 c_wait_stats; json = {"x,y": {"p1":..,"p2":..,"ws":..}}"""
import csv, json, sys
from collections import defaultdict

ZONES = [("TRISC_1", "c_pass1", "p1"), ("TRISC_1", "c_pass2", "p2"), ("TRISC_0", "c_wait_stats", "ws")]


def core_times(path, run):
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = [r for r in csv.DictReader(f, skipinitialspace=True) if r["run host ID"] == str(run)]
    k = 1.0 / mhz
    st = defaultdict(list)
    dur = defaultdict(lambda: defaultdict(float))
    want = {(r, z): key for r, z, key in ZONES}
    for r in rows:
        key = want.get((r["RISC processor type"], r["zone name"]))
        if key is None or r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            st[(c, key)].append(t)
        elif st[(c, key)]:
            dur[c][key] += (t - st[(c, key)].pop()) * k
    xs = sorted({c[0] for c in dur})
    ys = sorted({c[1] for c in dur})
    return {f"{xs.index(c[0])},{ys.index(c[1])}": dict(v) for c, v in dur.items()}


if __name__ == "__main__":
    out = core_times(sys.argv[1], sys.argv[2])
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=0)
    for k in sorted(out, key=lambda s: tuple(map(int, s.split(",")))[::-1]):
        print(k, {a: round(b) for a, b in out[k].items()})
