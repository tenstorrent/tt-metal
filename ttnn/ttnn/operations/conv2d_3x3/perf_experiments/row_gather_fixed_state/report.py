# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with this dir's runs.jsonl records of
the same RGFS_TAG (host only, no ttnn).

    python3 -I report.py <profiler dir or ops_perf_results csv> <tag>
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def find_csv(p):
    p = Path(p)
    if p.is_file():
        return p
    return max(p.glob("reports/*/ops_perf_results*.csv"), key=lambda q: q.stat().st_mtime)


def main():
    csv_path, tag = find_csv(sys.argv[1]), sys.argv[2]
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["OP CODE"].startswith("GenericOp")]
    recs = [json.loads(line) for line in open(HERE / "runs.jsonl")]
    recs = [r for r in recs if r.get("tag") == tag]
    # The precompile collect pass runs every body without dispatching: the real dispatches are the last records.
    recs = recs[len(recs) - len(rows) :] if len(recs) >= len(rows) else recs
    if len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} records")
    for rec, row in zip(recs, rows):
        ns = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        out = dict(tag=tag, case=rec["case"], variant=rec["variant"], ns=ns, pcc=rec["pcc"], ok=rec["ok"])
        for k in ("gather", "row_split", "fixed_capable", "bm", "bk", "c_in_p", "stride", "dilation"):
            out[k] = rec.get(k)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
