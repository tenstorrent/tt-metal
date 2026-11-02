# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Map the GenericOp rows of a profiler ops CSV onto (cell, variant) in dispatch order.

    python3 table.py <ops.csv> <cells comma list in CELLS order> <variants comma list>
"""
import csv
import sys

rows = [
    r
    for r in csv.DictReader(open(sys.argv[1]))
    if "GenericOp" in r.get("OP CODE", "") and r.get("DEVICE KERNEL DURATION [ns]")
]
cells = sys.argv[2].split(",")
variants = sys.argv[3].split(",")
ns = [int(float(r["DEVICE KERNEL DURATION [ns]"])) for r in rows]
assert len(ns) == len(cells) * len(variants), (len(ns), len(cells), len(variants))
print(f"{'cell':32s} " + " ".join(f"{v:>10s}" for v in variants) + "   ratio(last/first)")
for i, c in enumerate(cells):
    vals = ns[i * len(variants) : (i + 1) * len(variants)]
    print(f"{c:32s} " + " ".join(f"{x:>10d}" for x in vals) + f"   {vals[-1] / vals[0]:.3f}")
