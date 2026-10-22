# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Label the newest profiler CSV's GenericOp rows with the bench order log (the LAST N log lines are the profiled
run) and print one line per labelled dispatch: label, DEVICE KERNEL DURATION [ns].

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/bcast_srcb_reuse/report.py [--csv X] [--log Y]
"""

import argparse
import csv
import glob
import os
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log", default=str(HERE / "bench_order.txt"))
    a = ap.parse_args()
    path = a.csv or max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
    rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"]]
    labels = [l.strip() for l in open(a.log) if l.strip()][-len(rows) :]
    assert len(labels) == len(rows), (len(labels), len(rows))
    print(f"csv: {path}")
    for lab, r in zip(labels, rows):
        print(f"{lab:50s} {int(float(r['DEVICE KERNEL DURATION [ns]'])):>10d}")


if __name__ == "__main__":
    main()
