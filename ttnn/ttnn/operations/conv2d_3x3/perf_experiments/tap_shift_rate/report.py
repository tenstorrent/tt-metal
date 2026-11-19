# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with test_tsr.py's run log (no ttnn).

    python3 report.py <ops_perf_results.csv> <runs/<tag>.jsonl> [--append results.jsonl --tag TAG]

Bit-exactness: output sha vs the `base` (or `base_nc`) record of the same case in the same log."""

import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("log")
    ap.add_argument("--append", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r["OP CODE"].startswith("GenericOp")]
    recs = [json.loads(line) for line in open(a.log)]
    if len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
        recs = recs[-len(rows) :] if len(recs) > len(rows) else recs
    base = {}
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        if rec["variant"] in ("base", "base_z"):
            base.setdefault(rec["case"], rec)
    out = []
    for rec in recs:
        if "ns" not in rec:
            continue
        b = base.get(rec["case"])
        rel = f"{rec['ns'] / b['ns']:6.3f}x" if b else "   -   "
        same = "-" if not b else ("bit-identical" if rec["out_sha"] == b["out_sha"] else "DIFFERS")
        print(
            f"{rec['case']:<26} {rec['variant']:<8} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f} (>= {rec['pcc_threshold']})  {same}  shift={rec.get('tap_shift')}"
        )
        out.append(dict(rec, tag=a.tag, bit_identical=same))
    if a.append:
        with open(a.append, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
