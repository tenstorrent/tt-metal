# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print DEVICE KERNEL DURATION [ns] (and core count) per GenericOp row of an ops perf CSV (newest by default).

    python3 rows.py [csv] [label1 label2 ...]
"""
import csv
import glob
import os
import sys

args = sys.argv[1:]
if args and args[0].endswith(".csv"):
    path, labels = args[0], args[1:]
else:
    path = max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
    labels = args
rows = list(csv.reader(open(path)))
head = rows[0]
col = head.index("DEVICE KERNEL DURATION [ns]")
print(path)
for i, row in enumerate(r for r in rows[1:] if "GenericOp" in r[0]):
    label = labels[i] if i < len(labels) else ""
    print(f"{i:2d} {label:28s} cores {row[7]:>4s}  kernel_ns {row[col]:>8s}")
