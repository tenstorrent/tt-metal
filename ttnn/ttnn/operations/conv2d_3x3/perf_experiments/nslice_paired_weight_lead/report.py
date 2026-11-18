# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with test_lead_op.py's run log (no ttnn).

    python3 report.py <ops_perf_results.csv> [runs/<log>.jsonl]   (default: newest run log)

Appends the joined records to results.jsonl and prints each dispatch against the run's baseline of the same case.
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
        else max((HERE / "runs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    )
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["OP CODE"].startswith("GenericOp")]
    recs = [json.loads(line) for line in open(log_path)]
    # A precompile collect pass may run test bodies before the real pass: the real dispatches are the last ones.
    if len(recs) > len(rows):
        recs = recs[len(recs) - len(rows) :]
    elif len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    base = {}
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        rec["csv"] = str(csv_path)
        rec["log"] = log_path.name
        if rec["variant"] == "baseline":
            base.setdefault(rec["case"], rec)
    with open(HERE / "results.jsonl", "a") as out:
        for rec in recs:
            if "ns" not in rec:
                continue
            out.write(json.dumps(rec) + "\n")
            b = base.get(rec["case"])
            rel = f"{rec['ns'] / b['ns']:6.3f}x" if b else "   -   "
            same = "-" if not b else ("bit-identical" if rec["out_sha"] == b["out_sha"] else "DIFFERS")
            bl = rec.get("blocking", {})
            print(
                f"{rec['case']:<22} {rec['variant']:<10} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f} "
                f"(>= {rec['pcc_threshold']})  {same:<13} bm {bl.get('bm')} bk {bl.get('bk')} out_n {bl.get('out_n')} "
                f"K {bl.get('k_tiles')} nslice {bl.get('nslice')} lead {bl.get('lead')} walk {bl.get('walk')} "
                f"mb {bl.get('m_blocks')} wd {bl.get('weight_depth')} sd {bl.get('slice_depth')} "
                f"fp {bl.get('footprint')}/{bl.get('budget')}"
            )


if __name__ == "__main__":
    main()
