# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with the harness's run log (no ttnn).

    python3 report.py <ops_perf_results.csv> [runs/run_*.jsonl]   (default: newest run log)

DEVICE KERNEL DURATION [ns] is the metric; `rel` is against the same case's `base` dispatch in the same run.
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
    if len(recs) > len(rows):
        recs = recs[len(recs) - len(rows) :]
    elif len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    base = {}
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        if rec["variant"] in ("base", "base_kb"):
            base[(rec["case"], rec["variant"])] = rec["ns"]
    print(f"# {csv_path}\n# {log_path}")
    for rec in recs:
        if "ns" not in rec:
            continue
        b = base.get((rec["case"], "base_kb" if rec["variant"].endswith("_kb") else "base"))
        rel = f"{rec['ns'] / b:6.3f}x" if b else "   -   "
        bl = rec.get("blocking") or {}
        pcc = rec.get("pcc")
        print(
            f"{rec['case']:<28} {rec['variant']:<7} {rec['ns']:>9} ns {rel}"
            f"  pcc {pcc if pcc is None else round(pcc, 6)}  eq_base {rec.get('equal_base')}"
            f"{' eq_earlier ' + str(rec['equal_earlier']) if rec.get('equal_earlier') else ''}"
            f"  kcols {bl.get('k_cols')} T {bl.get('T')} out_n {bl.get('out_n')} Kcol {bl.get('k_col')}"
            f" bm {bl.get('bm')} bk {bl.get('bk')} nslice {int(bool(bl.get('nslice')))} mb {bl.get('mblocks')}"
            f"{'  ablate=' + rec['ablate'] if rec.get('ablate') else ''}"
        )


if __name__ == "__main__":
    main()
