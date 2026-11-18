# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with the harness's run log.

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
    # A precompile collect pass runs every test body once without the device (bogus outputs) before the real pass:
    # the real dispatches are the last len(rows) records.
    if len(recs) > len(rows):
        recs = recs[len(recs) - len(rows) :]
    elif len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    base = {}
    for rec, row in zip(recs, rows):
        ns = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        rec["ns"] = ns
        if rec["variant"] == "committed":
            base[rec["case"]] = ns
    for rec in recs:
        if "ns" not in rec:
            continue
        b = base.get(rec["case"])
        rel = f"{rec['ns'] / b:6.3f}x" if b else "   -   "
        print(
            f"{rec['case']:<20} {rec['variant']:<20} {rec['ns']:>9} ns  {rel}  pcc {rec['pcc']:.6f}  "
            f"relrms {rec['rel_rms']:.4f}  equal {rec['equal_committed']}  {rec.get('blocking')}"
        )


if __name__ == "__main__":
    main()
