#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core key events of the combine tail for one run (ns from the run's earliest kernel start).
usage: unit_timeline.py <profile_log_device.csv> <run_index> [core_x]
Columns: p1end (T1 c_pass1 end), coll (T2 c_collapse end = partials packed), gsend (BRISC w_gather_send end),
wgath (root BRISC w_wait_gather end), fin (T2 c_finalize end = root pack done), stats (T0 c_wait_stats end),
exp (T2 c_expand end), p2s (T1 c_pass2 start), end (T2 kernel end), aff (NCRISC r_ones_shift_affine end)."""
import csv, sys
from collections import defaultdict

path, idx = sys.argv[1], int(sys.argv[2])
cx = sys.argv[3] if len(sys.argv) > 3 else None
with open(path) as f:
    head = f.readline()
    mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
k = 1000.0 / mhz
runs = sorted({r["run host ID"] for r in rows}, key=int)
run = runs[idx]
rows = [r for r in rows if r["run host ID"] == run and r["type"] in ("ZONE_START", "ZONE_END")]
t0 = min(int(r["time[cycles since reset]"]) for r in rows if r["zone name"].endswith("KERNEL"))
ev = defaultdict(dict)
want = {
    ("TRISC_1", "c_pass1", "ZONE_END"): "p1end",
    ("TRISC_2", "c_collapse", "ZONE_END"): "coll",
    ("BRISC", "w_gather_send", "ZONE_END"): "gsend",
    ("BRISC", "w_wait_gather", "ZONE_END"): "wgath",
    ("TRISC_2", "c_finalize", "ZONE_END"): "fin",
    ("BRISC", "w_mcast_send", "ZONE_END"): "mcast",
    ("TRISC_0", "c_wait_stats", "ZONE_END"): "stats",
    ("TRISC_2", "c_expand", "ZONE_END"): "exp",
    ("TRISC_1", "c_pass2", "ZONE_START"): "p2s",
    ("TRISC_2", "TRISC-KERNEL", "ZONE_END"): "end",
    ("NCRISC", "r_ones_shift_affine", "ZONE_END"): "aff",
    ("TRISC_0", "c_wait_affine", "ZONE_END"): "waff",
}
for r in rows:
    key = (r["RISC processor type"], r["zone name"], r["type"])
    if key in want:
        c = (int(r["core_x"]), int(r["core_y"]))
        ev[c].setdefault(want[key], (int(r["time[cycles since reset]"]) - t0) * k)
cols = ["p1end", "coll", "gsend", "wgath", "fin", "mcast", "stats", "aff", "waff", "exp", "p2s", "end"]
print(f"run {run}  " + "core    " + " ".join(f"{c:>6s}" for c in cols))
for c in sorted(ev, key=lambda c: (c[0], c[1])):
    if cx is not None and str(c[0]) != cx:
        continue
    print(f"{'':10s}{str(c):8s}" + " ".join(f"{ev[c][n]:6.0f}" if n in ev[c] else "     -" for n in cols))
