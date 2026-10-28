# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""usage: python parse.py <report_dir> <labels.txt>  -> label, DEVICE KERNEL DURATION [ns] (GenericOp rows, in order)."""
import csv
import sys
from pathlib import Path

f = next(Path(sys.argv[1]).glob("ops_perf_results*.csv"))
rows = [r for r in csv.DictReader(open(f)) if "generic" in r["OP CODE"].lower()]
labels = open(sys.argv[2]).read().split()
if len(rows) != len(labels):
    print(f"WARNING rows={len(rows)} labels={len(labels)}")
rows = rows[len(rows) - len(labels) :]  # align from the end (earlier ops in the session are not labelled)
for lab, r in zip(labels, rows):
    print(f"{lab:40s} {r['DEVICE KERNEL DURATION [ns]']}")
