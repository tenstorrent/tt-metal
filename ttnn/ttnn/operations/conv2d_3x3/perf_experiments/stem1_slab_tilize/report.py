# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with this harness's run log (no ttnn).

    python3 report.py <ops_perf_results.csv> <runs/run_*.jsonl> [--append results.jsonl --tag TAG]

Bit-exactness is judged against the `baseline` variant of the same case in the same log (sha256 of the output).
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("log")
    ap.add_argument("--append", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r["OP CODE"].startswith("GenericOp")]
    recs = [json.loads(line) for line in open(args.log)]
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
    out = []
    for rec in recs:
        if "ns" not in rec:
            continue
        b = base.get(rec["case"])
        rel = f"{rec['ns'] / b['ns']:6.3f}x" if b else "   -   "
        same = "-" if not b else ("bit-identical" if rec["out_sha"] == b["out_sha"] else "DIFFERS")
        bl = rec.get("blocking", {})
        print(
            f"{rec['case']:<40} {rec['variant']:<9} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f} "
            f"(>= {rec['pcc_threshold']}) mae {rec.get('mae', 0):.5f} max {rec.get('max_abs', 0):.4f}  {same:<13} "
            f"bm {bl.get('bm')} bk {bl.get('bk')} sb {bl.get('sb')} fd {bl.get('face_direct')} sd {bl.get('sd')}"
        )
        out.append(dict(rec, tag=args.tag, csv=str(args.csv), bit_identical=same))
    if args.append:
        with open(args.append, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
