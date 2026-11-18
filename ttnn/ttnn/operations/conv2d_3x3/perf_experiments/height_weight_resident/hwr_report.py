# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pairs a profiled test_hwr_perf session with its records (no device, no ttnn import).

    python3 hwr_report.py <ops_perf_results csv> <HWR_TAG>

GenericOp rows of the CSV are the conv2d_3x3 dispatches in call order; the last len(rows) perf records with the tag
are the real pass (the precompile collect pass writes records too). Copies profile_log_device.csv next to the CSV into
runs/<tag>_device.csv.gz, prints per dispatch: ns, PCC / mean abs error vs a fp32 torch conv, the blocking, and the
weight slack of the compute cores (weight_slack.py: how long the matmul of each K-block waited on its round).
"""

import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    csv_path, tag = Path(sys.argv[1]), sys.argv[2]
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[header.index("OP CODE")].startswith("GenericOp")]
    col = header.index("DEVICE KERNEL DURATION [ns]")
    recs = [json.loads(line) for line in open(HERE / "results.jsonl")]
    recs = [r for r in recs if r.get("kind") == "perf" and r.get("tag") == tag]
    if len(recs) >= len(rows):
        recs = recs[len(recs) - len(rows) :]
    else:
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} records")
    dev_log = csv_path.parent / "profile_log_device.csv"
    if dev_log.exists():
        (HERE / "runs").mkdir(exist_ok=True)
        with open(dev_log, "rb") as src, gzip.open(HERE / "runs" / f"{tag}_device.csv.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    out = []
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row[col]))
        out.append(rec)
        blk = " ".join(
            f"{k} {rec.get(k)}" for k in ("bm", "bk", "nb", "nkb", "wd", "w_pages", "im2col", "center", "ring", "kwf")
        )
        print(
            f"{rec['case']:<16} {rec['variant']:<8} {rec['ns']:>9} ns  pcc {rec['pcc']:.6f} mae {rec['mae']:.4f}  {blk}"
        )
    with open(HERE / "results_perf.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(dict(rec, csv=str(csv_path))) + "\n")


if __name__ == "__main__":
    main()
