# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Label the newest ops_perf_results CSV's GenericOp rows with the LAST N lines of gm_order.txt (the precompile
pass logs labels too) and print DEVICE KERNEL DURATION [ns] per (case, variant) plus the ratio vs `off`.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/gamma_mcast/gm_report.py [--csv X] [--log Y]
"""

import argparse
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
ap.add_argument("--log", default=os.path.join(HERE, "gm_order.txt"))
a = ap.parse_args()
path = a.csv or max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"]]
labels = [l.strip() for l in open(a.log) if l.strip()][-len(rows) :]
assert len(labels) == len(rows), (len(labels), len(rows))
print(f"csv: {path}")
base = {}
for lab, r in zip(labels, rows):
    f = lab.split("|")
    case, var = f[0], f[1]
    ns = int(float(r["DEVICE KERNEL DURATION [ns]"]))
    if var == "off":
        base[case] = ns
    ratio = f"{base[case] / ns:.3f}x" if case in base else "-"
    print(f"{case:16s} {var:8s} {ns:>10d} ns  {ratio:>7s}  {'|'.join(f[2:])}")
