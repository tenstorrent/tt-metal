# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-run, per-core zone summary of a profiled tile_fill_split_nocs session (no device, no ttnn import).

    python3 tfs_zones.py <report dir or profile_log_device.csv> [--zones reader_fill,compute_untilize,...]

For each run (one dispatch on >= 8 cores): the op span (first kernel start .. last kernel end, us), then for each
zone: max / p50 over cores of the per-core sum, the core holding the max, and the same zone on the cores of the
grid's last physical column (the cores whose next shard sits on the next grid row).
Also prints DEVICE KERNEL DURATION per op from the ops_perf_results CSV in the same dir.
"""

import argparse
import collections
import csv
import glob
import os
import statistics
import sys

DEFAULT_ZONES = [
    "reader_fill",
    "reader_fill_reserve",
    "reader_fill_fwd_wait",
    "compute_untilize",
    "writer_fill",
    "writer_weight_reserve",
    "writer_weight_fetch",
    "TRISC-KERNEL",
    "BRISC-KERNEL",
    "NCRISC-KERNEL",
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--zones", default=",".join(DEFAULT_ZONES))
    ap.add_argument("--grid", default="", help="comma-separated zones to print as a per-core grid (y rows, x columns)")
    args = ap.parse_args()
    path = args.path
    report_dir = path if os.path.isdir(path) else os.path.dirname(path)
    log = os.path.join(path, "profile_log_device.csv") if os.path.isdir(path) else path
    for csv_path in glob.glob(os.path.join(report_dir, "ops_perf_results*.csv")):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r.get("DEVICE KERNEL DURATION [ns]"):
                    print(
                        f"op {r.get('OP CODE', '?')[:30]:30s} DEVICE KERNEL DURATION {r['DEVICE KERNEL DURATION [ns]']:>10} ns  cores {r.get('CORE COUNT')}"
                    )
    freq, runs = load(log)
    zones = args.zones.split(",")
    for run_id in sorted(runs):
        ev = runs[run_id]
        cores = {c for c, *_ in ev}
        if len(cores) < 8:
            continue
        stacks = collections.defaultdict(list)
        sums = collections.defaultdict(float)
        t_lo, t_hi = None, None
        for core, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
            if zone.endswith("-KERNEL"):
                t_lo = t if t_lo is None else min(t_lo, t)
                t_hi = t if t_hi is None else max(t_hi, t)
            key = (core, zone)
            if kind == "ZONE_START":
                stacks[key].append(t)
            elif kind == "ZONE_END" and stacks[key]:
                sums[key] += t - stacks[key].pop()
        max_x = max(c[0] for c in cores)
        print(f"=== run {run_id}: {len(cores)} cores, span {(t_hi - t_lo) / freq:.2f} us, last column x={max_x}")
        print(
            f"  {'zone':<24}{'max':>8}{'p50':>8}  {'max core':<10}{'lastcol max':>12}{'lastcol p50':>12}{'rest max':>9}"
        )
        for z in zones:
            per = {c: sums[(c, z)] / freq for c in cores if (c, z) in sums}
            if not per:
                continue
            vals = list(per.values())
            mc = max(per, key=per.get)
            last = [v for c, v in per.items() if c[0] == max_x]
            rest = [v for c, v in per.items() if c[0] != max_x]
            print(
                f"  {z:<24}{max(vals):8.2f}{statistics.median(vals):8.2f}  {str(mc):<10}"
                f"{(max(last) if last else 0):12.2f}{(statistics.median(last) if last else 0):12.2f}"
                f"{(max(rest) if rest else 0):9.2f}"
            )
        for z in [g for g in args.grid.split(",") if g]:
            per = {c: sums[(c, z)] / freq for c in cores if (c, z) in sums}
            xs = sorted({c[0] for c in cores})
            ys = sorted({c[1] for c in cores})
            print(f"  grid {z} (us; TRISC zones are summed over the 3 TRISCs)")
            print("     y\\x " + "".join(f"{x:>6}" for x in xs))
            for y in ys:
                print(f"  {y:>6} " + "".join(f"{per[(x, y)]:6.1f}" if (x, y) in per else "     ." for x in xs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
