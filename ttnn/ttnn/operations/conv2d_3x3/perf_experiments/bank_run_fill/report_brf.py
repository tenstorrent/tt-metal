# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE KERNEL DURATION per `brf|...` signpost window: python3 report_brf.py <ops_perf_results.csv> [...]"""

import csv
import sys

for path in sys.argv[1:]:
    label, acc = None, 0.0
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("OP TYPE") == "signpost":
                if label and acc > 0:
                    print(f"{label}\t{acc / 1000:.1f} us")
                head = row["OP CODE"]
                label, acc = (head[4:], 0.0) if head.startswith("brf|") else (None, 0.0)
                continue
            if label and row.get("DEVICE KERNEL DURATION [ns]"):
                acc += float(row["DEVICE KERNEL DURATION [ns]"])
    if label and acc > 0:
        print(f"{label}\t{acc / 1000:.1f} us")
