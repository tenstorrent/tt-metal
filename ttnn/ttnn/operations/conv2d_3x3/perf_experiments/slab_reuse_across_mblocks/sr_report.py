# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pairs a profiled test_sr_perf session with its test order (no device, no ttnn import).

    python3 sr_report.py <report dir> <SR_TAG>

Reads the session's ops_perf_results CSV (conv2d_3x3 dispatches = GenericOp rows, in dispatch order; column 20 is
DEVICE KERNEL DURATION [ns]) and the results.jsonl perf records written with SR_TAG=<tag> (same order), and prints
per dispatch: case, variant, ns, PCC, slab rows, core_slab, and from the device log the per-core zone sums (us, max /
p50 over cores) of compute_untilize (TRISC zone / 3), reader_fill and writer_fill.
"""

import collections
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import zone_report  # noqa: E402

RESULTS = os.path.join(HERE, "results.jsonl")
ZONES = [("TRISC", "compute_untilize", 3.0), ("NCRISC", "reader_fill", 1.0), ("BRISC", "writer_fill", 1.0)]


def per_run_zones(log):
    freq, runs = zone_report.load(log)
    out = []
    for run_id in sorted(runs):
        ev = runs[run_id]
        if len({c for c, *_ in ev}) < 8:
            continue
        stacks = collections.defaultdict(list)
        sums = collections.defaultdict(float)
        for core, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
            key = (core, risc, zone)
            if kind == "ZONE_START":
                stacks[key].append(t)
            elif kind == "ZONE_END" and stacks[key]:
                sums[key] += t - stacks[key].pop()
        z = {}
        for prefix, zone, div in ZONES:
            per = collections.defaultdict(float)
            for (core, risc, name), cyc in sums.items():
                if name == zone and risc.startswith(prefix):
                    per[core] += cyc / freq / div
            vals = list(per.values())
            z[zone] = (max(vals), statistics.median(vals)) if vals else (0.0, 0.0)
        out.append(z)
    return out


def main(report_dir, tag):
    (csv_path,) = glob.glob(os.path.join(report_dir, "ops_perf_results*.csv"))
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[19].strip() == "DEVICE KERNEL DURATION [ns]", header[19]
        op_col = header.index("OP CODE")
        rows = [r for r in reader if "GenericOp" in r[op_col]]
    recs = [json.loads(l) for l in open(RESULTS)]
    recs = [r for r in recs if r.get("kind") == "perf" and r.get("tag") == tag]
    log = os.path.join(report_dir, "profile_log_device.csv")
    zones = per_run_zones(log) if os.path.exists(log) else [{} for _ in rows]
    if not (len(rows) == len(recs) == len(zones)):
        print(f"WARNING: {len(rows)} dispatches, {len(recs)} records, {len(zones)} zoned runs (zones dropped)")
        if len(zones) != len(rows):
            zones = [{} for _ in rows]
    print(
        f"{'case':<40}{'variant':<12}{'ns':>9}{'pcc':>9}{'rows':>6}{'core':>6}"
        + "".join(f"{z[1][:14]:>16}" for z in ZONES)
    )
    for row, rec, z in zip(rows, recs, zones):
        zs = "".join(f"{z.get(name, (0, 0))[0]:8.2f}/{z.get(name, (0, 0))[1]:<7.2f}" for _, name, _ in ZONES)
        print(
            f"{rec['case']:<40}{rec['variant']:<12}{row[19]:>9}{rec['pcc']:9.5f}{rec['slab_rows']:>6}"
            f"{str(rec['core_slab']):>6} {zs}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
