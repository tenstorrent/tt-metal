#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print `DEVICE KERNEL DURATION [ns]` per dispatch from the newest profiler
report, one line per row, in dispatch order (which is the order
`test_write_issue_menu.py` documents). Host-only; no device.

    python3 .../write_issue_menu/read_perf.py [label]
"""
import csv
import glob
import os
import sys

reports = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
if not reports:
    sys.exit("no ops_perf_results*.csv under generated/profiler/reports")
path = reports[-1]
label = sys.argv[1] if len(sys.argv) > 1 else ""
rows = list(csv.DictReader(open(path)))
key = "DEVICE KERNEL DURATION [ns]"
brisc = "DEVICE BRISC KERNEL DURATION [ns]"
ncrisc = "DEVICE NCRISC KERNEL DURATION [ns]"
print(f"# {label}  <- {path}")
vals = []
for i, r in enumerate(rows):
    if key not in r or not r[key].strip():
        continue
    shape = f"{r.get('INPUT_0_Z','?')}x{r.get('INPUT_0_Y','?')}x{r.get('INPUT_0_X','?')}"
    b = r.get(brisc, "").strip() or "-"
    n = r.get(ncrisc, "").strip() or "-"
    print(f"  row{i:02d} {shape:>16s}  kernel={r[key]:>8s}  brisc={b:>8s}  ncrisc={n:>8s}")
    vals.append(int(r[key]))
if vals:
    print(f"  n={len(vals)}")

# --- `--ab MODE,MODE,... REPS`: decode the interleaved test's row order ------
if "--ab" in sys.argv:
    import statistics

    i = sys.argv.index("--ab")
    modes = sys.argv[i + 1].split(",")
    reps = int(sys.argv[i + 2])
    n = len(modes) * reps
    for phase, off in (("WRITE-STAGE", 0), ("WHOLE-OP", n)):
        if len(vals) < off + n:
            continue
        print(f"\n  === {phase} (paired, {reps} reps interleaved) ===")
        base = None
        for m_i, m in enumerate(modes):
            v = [vals[off + r * len(modes) + m_i] for r in range(reps)]
            med = statistics.median(v)
            if base is None:
                base = med
            print(f"    {m:14s} med={med:8.0f}  ({100*(med-base)/base:+5.1f}% vs {modes[0]})  reps={v}")
