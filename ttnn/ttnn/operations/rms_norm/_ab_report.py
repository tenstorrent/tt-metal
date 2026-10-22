# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Median old/new table for test_rms_norm_perf_guard_ab.py (labels `case|old`, `case|new2`, ...) off the newest
ops_perf_results CSV: python3 -m ttnn.operations.rms_norm._ab_report [--csv <csv>] [--log <labels>]"""

import argparse
import statistics
from collections import defaultdict

from ttnn.operations.rms_norm._perf_report import durations, newest_ops_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log", default="generated/rms_norm_bench_order.txt")
    a = ap.parse_args()
    rows = durations(a.csv or newest_ops_csv())
    labels = [l.strip() for l in open(a.log) if l.strip()][-len(rows) :]
    by = defaultdict(lambda: defaultdict(list))
    order = []
    for lab, r in zip(labels, rows):
        case, impl = lab.split("|")
        if case not in order:
            order.append(case)
        by[case][impl.rstrip("0123456789")].append(float(r["DEVICE KERNEL DURATION [ns]"]))
    print(f"{'case':<20} {'old med':>10} {'new med':>10} {'new/old':>8}   old runs / new runs")
    for c in order:
        a_name = next((k for k in ("old", "gmoff", "holdoff") if k in by[c]), None)
        b_name = "new"
        if a_name is None:
            continue
        o, n = by[c][a_name], by[c][b_name]
        mo, mn = statistics.median(o), statistics.median(n)
        print(f"{c:<20} {mo:>10.0f} {mn:>10.0f} {mn / mo:>8.3f}   {[int(v) for v in o]} / {[int(v) for v in n]}")


if __name__ == "__main__":
    main()
