# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pairs a profiled sweep session with its test order (no device, no ttnn import).

    python3 tfs_report.py <report dir> <TFS_TAG>

Reads the session's ops_perf_results CSV (conv2d_3x3 dispatches = GenericOpDeviceOperation rows, in dispatch order)
and the results.jsonl perf records written with TFS_TAG=<tag> (same order), and prints per dispatch: case, variant,
DEVICE KERNEL DURATION, PCC, and from the device log the slowest core's reader_fill / compute_untilize (us; the TRISC
zone divided by 3 TRISCs) and the max over cores of the reader's wait on the writer's fill half.
"""

import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import tfs_zones  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results.jsonl")


def per_run_zones(log):
    freq, runs = tfs_zones.load(log)
    out = []
    for run_id in sorted(runs):
        ev = runs[run_id]
        cores = {c for c, *_ in ev}
        if len(cores) < 8:
            continue
        stacks = collections.defaultdict(list)
        sums = collections.defaultdict(float)
        for core, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
            key = (core, risc, zone)
            if kind == "ZONE_START":
                stacks[key].append(t)
            elif kind == "ZONE_END" and stacks[key]:
                sums[key] += t - stacks[key].pop()

        def zmax(risc_prefix, zone, div=1.0):
            per = collections.defaultdict(float)
            for (core, risc, z), cyc in sums.items():
                if z == zone and risc.startswith(risc_prefix):
                    per[core] += cyc / freq / div
            return max(per.values()) if per else 0.0

        out.append(
            dict(
                fill=zmax("NCRISC", "reader_fill"),
                fwd_wait=zmax("NCRISC", "reader_fill_fwd_wait"),
                untilize=zmax("TRISC", "compute_untilize", 3.0),
            )
        )
    return out


def main(report_dir, tag):
    (csv_path,) = glob.glob(os.path.join(report_dir, "ops_perf_results*.csv"))
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if "GenericOp" in r.get("OP CODE", "")]
    recs = [json.loads(l) for l in open(RESULTS)]
    recs = [r for r in recs if r.get("kind") == "perf" and r.get("tag") == tag]
    zones = per_run_zones(os.path.join(report_dir, "profile_log_device.csv"))
    if not (len(rows) == len(recs) == len(zones)):
        print(f"WARNING: {len(rows)} dispatches, {len(recs)} records, {len(zones)} zoned runs")
    print(f"{'case':<22}{'variant':<16}{'ns':>10}{'pcc':>9}{'fill max':>10}{'until max':>10}{'fwd max':>9}")
    for row, rec, z in zip(rows, recs, zones):
        print(
            f"{rec['case']:<22}{rec['variant']:<16}{row['DEVICE KERNEL DURATION [ns]']:>10}{rec['pcc']:9.5f}"
            f"{z['fill']:10.2f}{z['untilize']:10.2f}{z['fwd_wait']:9.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
