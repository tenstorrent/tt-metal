#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per signpost window "<case>|<variant>|start" .. "|end" of an ops_perf_results CSV: the device kernel
time of every launch, and the median of launches 2..N (launch 1 dropped)."""

import csv
import statistics
import sys

import os

KEY = "DEVICE KERNEL DURATION [ns]"
VERBOSE = os.environ.get("G6TU_VERBOSE") == "1"


def main(path, skip_first=1):
    rows = list(csv.DictReader(open(path)))
    window = None
    groups = []
    for row in rows:
        if "signpost" in (row.get("OP TYPE") or ""):
            name = row.get("OP CODE") or ""
            if name.endswith("|start"):
                window = (name[: -len("|start")], [], [], [])
            elif name.endswith("|end") and window is not None:
                groups.append(window)
                window = None
            continue
        if window is not None and row.get(KEY):
            window[1].append(float(row[KEY]))
            window[2].append(row.get("OP CODE"))
            window[3].append(row)
    for tag, values, ops, launch_rows in groups:
        timed = values[skip_first:] if len(values) > skip_first else values
        med = statistics.median(timed) if timed else float("nan")
        line = f"{tag:40s} median={med / 1000:9.2f} us  n={len(values)}  all={[round(v / 1000, 2) for v in values]}"
        if VERBOSE and launch_rows:
            r = launch_rows[-1]

            def us(col):
                v = r.get(col) or ""
                return f"{float(v) / 1000:.2f}" if v else "-"

            line += (
                f"  cores={r.get('CORE COUNT')} BR={us('DEVICE BRISC KERNEL DURATION [ns]')}"
                f" NC={us('DEVICE NCRISC KERNEL DURATION [ns]')} T0={us('DEVICE TRISC0 KERNEL DURATION [ns]')}"
                f" T1={us('DEVICE TRISC1 KERNEL DURATION [ns]')} T2={us('DEVICE TRISC2 KERNEL DURATION [ns]')}"
            )
        else:
            line += f"  ops={sorted(set(ops))}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
