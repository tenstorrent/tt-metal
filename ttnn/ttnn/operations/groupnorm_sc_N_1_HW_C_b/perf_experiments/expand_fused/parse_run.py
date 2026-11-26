#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair one profiled bench_expand.py run with its results.jsonl records (host-only).

usage: parse_run.py <report_dir> <n_last_records> [zone,zone,...]

Rows of ops_perf_results (GenericOpDeviceOperation, in dispatch order) pair with the last N records of
results.jsonl. For each dispatch: DEVICE KERNEL DURATION [ns], the TRISC_1 (math) per-core sum of each zone
(p50 / max over cores, ns), PCC / rel-RMS.
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))


def main(report, n_last, zones):
    ops_csv = glob.glob(os.path.join(report, "ops_perf_results*.csv"))[0]
    rows = [r for r in csv.DictReader(open(ops_csv)) if r["OP CODE"] == "GenericOpDeviceOperation"]
    # The precompile collect pass also runs every body (no dispatch): pair the op rows with the LAST records.
    recs = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))][-len(rows) :]
    with open(os.path.join(report, "profile_log_device.csv")) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        dev = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    runs = sorted({r["run host ID"] for r in dev}, key=int)
    stacks = defaultdict(list)
    dur = defaultdict(lambda: defaultdict(float))  # (run, risc, zone) -> core -> ns
    last_end = {}  # (run, risc, zone, core) -> end cycle of its last execution
    for r in dev:
        if r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        name = r["zone name"]
        if name.endswith("-KERNEL") or name.endswith("-FW"):
            continue
        key = (r["run host ID"], r["RISC processor type"], name, (r["core_x"], r["core_y"]))
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            stacks[key].append(t)
        elif stacks[key]:
            s0 = stacks[key].pop()
            dur[key[:3]][key[3]] += (t - s0) * k
            last_end[key] = t
    if len(rows) != len(recs):
        print(f"WARNING: {len(rows)} op rows vs {len(recs)} records")
    hdr = f"{'case':26s} {'variant':10s} {'affine':10s} {'kernel_ns':>10s}"
    for z in zones:
        hdr += f" {z + ' p50/max':>22s}"
    print(hdr + "   pcc       rms")
    for i, (row, rec) in enumerate(zip(rows, recs)):
        run = runs[i] if i < len(runs) else None
        line = (
            f"{rec['case']:26s} {rec['variant']:10s} {rec['affine']:10s} {int(row['DEVICE KERNEL DURATION [ns]']):10d}"
        )
        for z in zones:
            risc, zn = {"@": ("TRISC_0", z[1:]), "%": ("TRISC_2", z[1:]), "^": ("NCRISC", z[1:])}.get(
                z[0], ("TRISC_1", z)
            )
            if z == "EXPAND_TRUE":
                # Stats landed (unpack thread leaves its wait) -> the packer finished the expand (a, b' published).
                v = [
                    (last_end[(run, "TRISC_2", "c_expand", c)] - last_end[(run, "TRISC_0", "c_wait_stats", c)]) * k
                    for c in dur[(run, "TRISC_2", "c_expand")]
                    if (run, "TRISC_0", "c_wait_stats", c) in last_end
                ]
            else:
                v = list(dur[(run, risc, zn)].values())
            line += f" {median(v):10.0f}/{max(v):10.0f}" if v else f" {'-':>22s}"
        print(line + f"   {rec['pcc']:.6f} {rec['rms']:.5f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else ["c_expand"])
