#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join the harness's `DISPATCH n | label` lines to the profiler CSV.

    python3 .../writer_starvation/parse.py <pytest_stdout.log> [ops_perf_results.csv]

Prints, per (shape, variant), every rep's DEVICE KERNEL DURATION [ns] and the
median, plus the ratio to that shape's `baseline` median.
"""
import csv
import glob
import os
import re
import statistics
import sys

LINE = re.compile(r"^DISPATCH (\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|")


def newest_csv():
    hits = glob.glob("generated/profiler/reports/*/ops_perf_results*.csv")
    if not hits:
        sys.exit("no ops_perf_results*.csv under generated/profiler/reports/")
    return max(hits, key=os.path.getmtime)


def main():
    log = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else newest_csv()
    labels = {}
    for line in open(log, errors="ignore"):
        m = LINE.match(line.strip())
        if m:
            labels[int(m.group(1))] = (m.group(4).strip(), m.group(3).strip(), m.group(2).strip())

    rows = list(csv.DictReader(open(csv_path)))
    key = "DEVICE KERNEL DURATION [ns]"
    durations = [int(r[key]) for r in rows if r.get(key) and r[key].strip().isdigit()]
    print(f"csv={csv_path}  dispatches_logged={len(labels)}  durations={len(durations)}")
    if len(durations) != len(labels):
        print("  !! COUNT MISMATCH — check for non-bench dispatches in the CSV")

    per = {}
    for i, ns in enumerate(durations, start=1):
        if i not in labels:
            continue
        shape, variant, _rep = labels[i]
        per.setdefault((shape, variant), []).append(ns)

    shapes = []
    for shape, _v in per:
        if shape not in shapes:
            shapes.append(shape)
    for shape in shapes:
        base = per.get((shape, "baseline"))
        base_med = statistics.median(base) if base else None
        print(f"\n=== {shape} ===")
        for (s, variant), v in per.items():
            if s != shape:
                continue
            med = statistics.median(v)
            rel = f"{base_med/med:6.3f}x" if base_med else "   n/a"
            print(f"  {variant:22s} median {med:8.0f} ns  {rel}   reps {v}")


if __name__ == "__main__":
    main()
