# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with the harness's run log (no ttnn).

    python3 report.py <ops_perf_results.csv> [runs/run_*.jsonl]   (default: newest run log)
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    csv_path = Path(sys.argv[1])
    log_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else max((HERE / "runs").glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    )
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["OP CODE"].startswith("GenericOp")]
    recs = [json.loads(line) for line in open(log_path)]
    # A precompile pass may run every body once without real device results: the real dispatches are the last rows.
    if len(recs) > len(rows):
        recs = recs[len(recs) - len(rows) :]
    elif len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    base = {}
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        if rec["variant"] == "d2":
            base[rec["case"]] = rec["ns"]
    for rec in recs:
        if "ns" not in rec:
            continue
        b = base.get(rec["case"])
        rel = f"{rec['ns'] / b:6.3f}x" if b else "   -   "
        bl = rec.get("blocking") or {}
        print(
            f"{rec['case']:<42} {rec['variant']:<5} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f}  eq {rec['equal_d2']!s:<5}"
            f"  bm {bl.get('bm')} bk {bl.get('bk')} nk {bl.get('nk')} mb {bl.get('mblocks')} depth {bl.get('depth')}"
            f" tm {int(bool(bl.get('tap_major')))} rs {int(bool(bl.get('row_split')))} fd {int(bool(bl.get('face_direct')))}"
            f" ns {int(bool(bl.get('nslice')))} slack {(bl.get('budget', 0) - bl.get('footprint', 0)) // 1024} KB"
        )


if __name__ == "__main__":
    main()
