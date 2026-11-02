"""Print DEVICE KERNEL DURATION [ns] of every GenericOp row of an ops_perf_results CSV."""

import csv
import sys

for row in csv.DictReader(open(sys.argv[1])):
    if "Generic" in row["OP CODE"]:
        print("KERNEL_NS", row["DEVICE KERNEL DURATION [ns]"])
