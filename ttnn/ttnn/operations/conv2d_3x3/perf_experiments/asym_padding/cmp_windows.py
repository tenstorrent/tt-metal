"""Median device kernel ns per `cmp|<case>|<op>|<i>` signpost window of a Tracy ops_perf_results CSV.

Usage: python cmp_windows.py <ops_perf_results.csv> [<csv> ...]
A window sums DEVICE KERNEL DURATION over the ops between `cmp|label|i` and the next signpost; empty windows (the
precompile collect pass runs the bodies without the device) are dropped.
"""

import csv
import statistics
import sys
from collections import defaultdict


def windows(path):
    sums = defaultdict(list)
    label, acc = None, 0
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("OP TYPE") == "signpost":
                if label is not None and acc > 0:
                    sums[label].append(acc)
                head = row["OP CODE"]
                label, acc = (head.rsplit("|", 1)[0][4:], 0) if head.startswith("cmp|") else (None, 0)
                continue
            if label is not None and row.get("DEVICE KERNEL DURATION [ns]"):
                acc += float(row["DEVICE KERNEL DURATION [ns]"])
    if label is not None and acc > 0:
        sums[label].append(acc)
    return sums


for path in sys.argv[1:]:
    for label, vals in windows(path).items():
        print(f"{label}\t{statistics.median(vals):.0f}\t{[round(v) for v in vals]}")
