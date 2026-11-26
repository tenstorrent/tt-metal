#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""One row per run host ID: kernel span and the p50-over-cores per-core-sum of selected zones (ns).

usage: zone_table.py <profile_log_device.csv> [label,label,...]   (labels in run order, optional)
Columns: span = TRISC_1 kernel span p50; math (TRISC_1) zones; wait zones on the thread that blocks (TRISC_0).
"""
import csv
import sys
from collections import defaultdict
from statistics import median

COLS = [
    ("TRISC_0", "c_wait_map"),
    ("TRISC_0", "c_p1_wait_x"),
    ("TRISC_1", "c_tilize"),
    ("TRISC_1", "c_p1_xk_chain"),
    ("TRISC_1", "c_p1_reduce"),
    ("TRISC_1", "c_pass1"),
    ("TRISC_1", "c_collapse"),
    ("TRISC_0", "c_wait_stats"),
    ("TRISC_1", "c_finalize"),
    ("TRISC_1", "c_expand"),
    ("TRISC_1", "c_p2_normalize"),
    ("TRISC_1", "c_untilize"),
    ("TRISC_1", "c_pass2"),
    ("NCRISC", "r_block"),
]


def main(path, labels):
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    stacks = defaultdict(list)
    dur = defaultdict(lambda: defaultdict(float))
    span = defaultdict(dict)
    for r in rows:
        if r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        run, core, risc, name = r["run host ID"], (r["core_x"], r["core_y"]), r["RISC processor type"], r["zone name"]
        t = int(r["time[cycles since reset]"])
        if name == "TRISC-KERNEL" or name.endswith("-KERNEL"):
            span[(run, risc)].setdefault(core, [0, 0])[0 if r["type"] == "ZONE_START" else 1] = t
            continue
        key = (run, risc, name, core)
        if r["type"] == "ZONE_START":
            stacks[key].append(t)
        elif stacks[key]:
            dur[(run, risc, name)][core] += (t - stacks[key].pop()) * k
    runs = sorted({r[0] for r in dur}, key=int)
    hdr = ["run", "span"] + [c[1].replace("c_", "").replace("r_", "r.") for c in COLS]
    print(" ".join(f"{h:>10s}" for h in hdr))
    for i, run in enumerate(runs):
        sp = [(e - s) * k for s, e in span[(run, "TRISC_1")].values() if e > s]
        vals = [median(sp) if sp else 0]
        for risc, name in COLS:
            v = list(dur[(run, risc, name)].values())
            vals.append(median(v) if v else float("nan"))
        lab = labels[i] if i < len(labels) else run
        print(f"{lab[:10]:>10s} " + " ".join(f"{v:10.0f}" for v in vals))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else [])
