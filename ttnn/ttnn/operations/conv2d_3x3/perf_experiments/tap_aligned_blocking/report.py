# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join a profiled run's ops_perf_results CSV with the variants test_tap_aligned.py logged (no device, no ttnn).

    python3 -I report.py <ops_perf_results_*.csv> [--tag NAME]

The CSV's GenericOpDeviceOperation rows are the variants in execution order (one dispatch each); they are matched
to the LAST N runs.jsonl records (a precompile collect pass may log extra records before them). With --tag, the
joined rows are appended to results.jsonl under that tag.
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tag")
    args = ap.parse_args()
    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if "GenericOp" in r["OP CODE"]]
    logs = [json.loads(line) for line in open(HERE / "runs.jsonl")]
    logs = logs[-len(rows) :] if rows else []
    if len(logs) != len(rows):
        raise SystemExit(f"{len(rows)} generic-op rows but {len(logs)} log records")
    joined = []
    for r, rec in zip(rows, logs):
        rec = dict(rec, kernel_ns=int(r["DEVICE KERNEL DURATION [ns]"]), cores=int(r["CORE COUNT"]), csv=args.csv)
        joined.append(rec)
        print(
            f"{rec['case']:22s} {rec['variant']:22s} bm {rec['bm']:2d} bk {rec['bk']:2d} kb {rec['num_k_blocks']:2d} "
            f"mb {rec['m_blocks']:2d} wd {rec['weight_depth']} ub {rec['ub']:2d} tap {int(rec['tap_major'])} "
            f"zones {int(rec['zones'])}  {rec['kernel_ns'] / 1000:8.1f} us  pcc {rec['pcc']:.5f} rms {rec['rel_rms']:.4f}"
        )
    if args.tag:
        with open(HERE / "results.jsonl", "a") as f:
            for rec in joined:
                f.write(json.dumps(dict(rec, tag=args.tag)) + "\n")


if __name__ == "__main__":
    main()
