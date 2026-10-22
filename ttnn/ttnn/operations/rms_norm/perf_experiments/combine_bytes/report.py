# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Label the newest profiler CSV's GenericOp rows with the bench order log (last N entries = the profiled pass).

    python3 report.py [--csv <ops_perf_results.csv>] [--log generated/combine_bytes_order.txt]
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--csv")
ap.add_argument("--log", default="generated/combine_bytes_order.txt")
a = ap.parse_args()
path = a.csv or max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"]]
labels = [l.strip() for l in open(a.log) if l.strip()]
labels = labels[-len(rows) :]
print(path, len(rows), "ops")
table = defaultdict(dict)
for lab, r in zip(labels, rows):
    case, variant = lab.split("|")[:2]
    ns = int(float(r["DEVICE KERNEL DURATION [ns]"]))
    table[case].setdefault(variant, []).append(ns)
    print(f"{lab:70s} {ns:>9d}")
print()
for case, v in table.items():
    base = v.get("base", [None])[0]
    cells = []
    for name, ns in v.items():
        s = f"{name}={','.join(map(str, ns))}"
        if base and name != "base":
            s += f" ({(ns[0] - base) / base * 100:+.1f}%)"
        cells.append(s)
    print(f"{case:20s} " + "  ".join(cells))
