# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair a profiled run's conv dispatches (ops CSV rows, in order) with the last N results.jsonl records and print
baseline vs candidate per case (no ttnn import).

    python3 report.py runs/<run>_ops.csv [--base baseline] [--cand ckb]
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--base", default="baseline")
    ap.add_argument("--cand", default="ckb")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.csv)) if r["OP CODE"] == "GenericOpDeviceOperation"]
    recs = [json.loads(l) for l in open(Path(__file__).parent / "results.jsonl")]
    recs = [r for r in recs if r.get("kind") != "vs_baseline"][-len(rows) :]
    by_case = {}
    for row, rec in zip(rows, recs):
        key = rec["case"] + (
            "/" + rec["layout"].split(".")[-1] + "/" + rec["dtype"].split(".")[-1] if "layout" in rec else ""
        )
        by_case.setdefault(key, {})[rec["variant"]] = (int(row["DEVICE KERNEL DURATION [ns]"]), rec)
    print(f"{'case':58s} {'base us':>8s} {'cand us':>8s} {'delta':>7s}  blocking base -> cand")
    for key, v in by_case.items():
        if args.base not in v or args.cand not in v:
            continue
        (b, rb), (c, rc) = v[args.base], v[args.cand]

        def blk(r):
            return f"bm{r.get('bm')} bk{r.get('bk')} nk{r.get('num_k_blocks')} mb{r.get('m_blocks')} ch{r.get('weight_chunks')} ckb{int(bool(r.get('ckb')))}"

        print(
            f"{key:58s} {b / 1000:8.1f} {c / 1000:8.1f} {100 * (c - b) / b:+6.1f}%  {blk(rb)} -> {blk(rc)}"
            f"  pcc {rb.get('pcc', 0):.5f}/{rc.get('pcc', 0):.5f}"
        )


if __name__ == "__main__":
    main()
