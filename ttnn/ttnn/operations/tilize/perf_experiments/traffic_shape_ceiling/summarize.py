#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Variant table for `traffic_shape_ceiling`: ns, GB/s, and per-RISC span mean/max.

Joins the newest profiler report to `logs/dispatch_order.json` (the driver
writes the exact dispatch order, and `ops_perf_results*.csv` is in dispatch
order), then prints per rung:

    device kernel ns (median over reps 1..N-1; rep 0 dropped as the warm rung)
    GB/s over the bytes THAT rung actually moves (read + write)
    NCRISC / BRISC per-core `*-KERNEL` span mean and max, and max/mean

The span mean/max come from the same `*-KERNEL` zone collector the sibling
`block_to_core_mapping/percore_map.py` uses (imported, not rewritten), so the
positional-gradient numbers are directly comparable to that idea's baseline
(op: NCRISC 5288/7588, BRISC 9710/12048). `--map` additionally prints the 8x8
grid map and the row/column correlations for one rung.

    python3 .../traffic_shape_ceiling/summarize.py [--map independent]
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "block_to_core_mapping"))
from percore_map import collect, corr, grid_maps  # noqa: E402

# Bytes each rung moves, per the traffic table in bench.py (64 cores x 16 KB).
SIDE_BYTES = 64 * 16384
MOVES = {
    "chained": ("rw", 2 * SIDE_BYTES),
    "chained_2k": ("rw", 2 * SIDE_BYTES),
    "independent": ("rw", 2 * SIDE_BYTES),
    "independent_2k": ("rw", 2 * SIDE_BYTES),
    "reads_only_512": ("r", SIDE_BYTES),
    "reads_only_2k": ("r", SIDE_BYTES),
    "writes_only": ("w", SIDE_BYTES),
}


def newest_report():
    dirs = sorted(glob.glob("generated/profiler/reports/*/"), key=os.path.getmtime)
    if not dirs:
        raise SystemExit("no generated/profiler/reports/* — run with --profile first")
    return dirs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    ap.add_argument("--map", default=None, help="print the 8x8 per-core grid map for this rung")
    args = ap.parse_args()

    rep_dir = args.report or newest_report()
    ops_csv = sorted(glob.glob(os.path.join(rep_dir, "ops_perf_results*.csv")))[0]
    dev_csv = os.path.join(rep_dir, "profile_log_device.csv")
    order = json.loads(open(os.path.join(HERE, "logs", "dispatch_order.json")).read())

    rows = list(csv.DictReader(open(ops_csv)))
    rows = [r for r in rows if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
    if len(rows) != len(order):
        print(f"WARNING: {len(rows)} profiled ops vs {len(order)} dispatches; taking the LAST {len(order)}")
        rows = rows[-len(order) :]

    st, _ = collect(dev_csv)
    runs = sorted({k[0] for k in st})
    if len(runs) >= len(order):
        runs = runs[-len(order) :]

    per_variant = {}
    for k, (row, rec) in enumerate(zip(rows, order)):
        ns = int(row["DEVICE KERNEL DURATION [ns]"])
        per_variant.setdefault(rec["variant"], []).append((rec["rep"], ns, runs[k] if k < len(runs) else None))

    hdr = f"{'rung':16s} {'ns(med)':>9s} {'all reps':>28s} {'GB/s':>7s} {'side':>5s}"
    for risc in ("NCRISC", "BRISC"):
        hdr += f" {risc + ' mean':>12s} {risc + ' max':>11s} {'m/m':>5s}"
    print(f"\nreport: {rep_dir}")
    print(hdr)
    for v in [x for x in MOVES if x in per_variant]:
        recs = per_variant[v]
        measured = [ns for rep, ns, _ in recs if rep > 0] or [ns for _, ns, _ in recs]
        med = statistics.median(measured)
        side, nbytes = MOVES[v]
        gbps = nbytes / med  # bytes/ns == GB/s
        line = f"{v:16s} {med:9.0f} {str(sorted(measured)):>28s} {gbps:7.1f} {side:>5s}"
        # per-core spans of the LAST measured rep of this rung
        run = [r for rep, _, r in recs if rep > 0 and r is not None]
        run = run[-1] if run else None
        for risc in ("NCRISC", "BRISC"):
            got = grid_maps(st, run, risc) if run is not None else None
            if not got:
                line += f" {'-':>12s} {'-':>11s} {'-':>5s}"
                continue
            cells = got[0]
            d = [c[0] for c in cells.values()]
            line += f" {statistics.mean(d):12.0f} {max(d):11.0f} {max(d)/statistics.mean(d):5.2f}"
        print(line)

    if args.map:
        recs = per_variant[args.map]
        run = [r for rep, _, r in recs if rep > 0 and r is not None][-1]
        print(f"\n=== per-core grid map, rung {args.map}, run host ID {run} ===")
        for risc in ("NCRISC", "BRISC"):
            got = grid_maps(st, run, risc)
            if not got:
                continue
            cells, _xs, _ys = got
            ys = sorted({y for _, y in cells})
            xs = sorted({x for x, _ in cells})
            print(f"\n  {risc} duration ns")
            print("      " + "".join(f"{x:>8d}" for x in xs))
            for y in ys:
                print(f"  y={y}  " + "".join(f"{cells[(x, y)][0]:8.0f}" for x in xs))
            print(f"  r(duration, grid ROW y)    = {corr([(y, cells[(x, y)][0]) for (x, y) in cells]):+.2f}")
            print(f"  r(duration, grid COLUMN x) = {corr([(x, cells[(x, y)][0]) for (x, y) in cells]):+.2f}")
            print(f"  r(end,      grid ROW y)    = {corr([(y, cells[(x, y)][2]) for (x, y) in cells]):+.2f}")


if __name__ == "__main__":
    sys.exit(main())
