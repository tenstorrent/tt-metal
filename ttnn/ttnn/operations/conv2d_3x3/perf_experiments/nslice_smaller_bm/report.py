# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with this harness's run log (no ttnn).

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
    # A precompile collect pass runs every test body once before the real pass: the real dispatches are the last ones.
    if len(recs) > len(rows):
        recs = recs[len(recs) - len(rows) :]
    elif len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    base = {}
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        if rec["variant"] == "baseline":
            base.setdefault(rec["case"], rec)
    for rec in recs:
        if "ns" not in rec:
            continue
        b = base.get(rec["case"])
        rel = f"{rec['ns'] / b['ns']:6.3f}x" if b else "   -   "
        same = "-" if not b else ("bit-identical" if rec["out_sha"] == b["out_sha"] else "differs")
        bl = rec.get("blocking", {})
        print(
            f"{rec['case']:<22} {rec['variant']:<10} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f} "
            f"(>= {rec['pcc_threshold']})  {same:<13} bm {bl.get('bm')} bk {bl.get('bk')} ub {bl.get('ub')} "
            f"nslice {bl.get('nslice')} alias {bl.get('alias')} sd {bl.get('slice_depth')} fp {bl.get('footprint')}"
        )


if __name__ == "__main__":
    main()
