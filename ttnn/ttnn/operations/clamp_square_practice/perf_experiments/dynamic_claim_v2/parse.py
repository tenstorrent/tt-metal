# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""usage: python3 parse.py <pytest_log> [report_dir]  -> DEVICE KERNEL DURATION per generic op, paired
in call order with the DC_RESULT lines of the log (latest report dir under TT_METAL_HOME by default)."""
import csv
import glob
import os
import sys

log = sys.argv[1]
rep = (
    sys.argv[2]
    if len(sys.argv) > 2
    else sorted(glob.glob("/localdev/jmitrovic/tt-metal/generated/profiler/reports/*/"))[-1]
)
rows = [l.split("info=")[0].strip() for l in open(log) if l.startswith("DC_RESULT")]
ops = list(csv.DictReader(open(glob.glob(f"{rep}/ops_perf_results*.csv")[0])))
ops = [o for o in ops if "generic" in o["OP CODE"].lower()]
print(f"report={rep} generic_ops={len(ops)} rows={len(rows)}")
rows = rows[-len(ops) :] if len(ops) <= len(rows) else rows
for o, r in zip(ops, rows):
    print(f"{int(o['DEVICE KERNEL DURATION [ns]']):>9} ns  {r}")
