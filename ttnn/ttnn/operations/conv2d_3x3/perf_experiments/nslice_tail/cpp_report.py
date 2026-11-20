# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join a run's cpp_device_perf_report.csv (device kernel duration per op, call order) with this harness's run log
(when the ops_perf CSV is missing). Pairs each variant with the baseline of the same case.
    python3 cpp_report.py <cpp_device_perf_report.csv> <runs/<tag>.jsonl>"""
import csv
import json
import sys

rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: int(r["GLOBAL CALL COUNT"]))
recs = [json.loads(l) for l in open(sys.argv[2])]
recs = recs[len(recs) - len(rows) :]
base = {}
for rec, row in zip(recs, rows):
    rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
    if rec["variant"] == "baseline":
        base[rec["case"]] = rec
for rec in recs:
    b = base.get(rec["case"])
    rel = f"{(rec['ns'] / b['ns'] - 1) * 100:+6.2f}%" if b and rec is not b else "   -   "
    same = "" if not b else ("bit-identical" if rec["out_sha"] == b["out_sha"] else "DIFFERS")
    print(f"{rec['case']:<26} {rec['variant']:<10} {rec['ns']:>8} ns {rel}  pcc {rec['pcc']:.6f}  {same}")
