# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with this harness's run log.

    python3 report.py <ops_perf_results.csv> <runs/run_*.jsonl> [--append results.jsonl --tag TAG]

Each row's ratio is against the `base` variant of the same case (`base_w` for a weights-ablated `_w` variant), the median
over its repeats. Bit-exactness and max abs diff are against base's saved output (runs/outs/<case>-base.pt).
"""

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent


def _maxdiff(case, variant):
    try:
        import torch
    except ImportError:
        return None
    a = HERE / "runs" / "outs" / f"{case.replace(':', '_')}-base.pt"
    b = HERE / "runs" / "outs" / f"{case.replace(':', '_')}-{variant}.pt"
    if not (a.exists() and b.exists()):
        return None
    x, y = torch.load(a).float(), torch.load(b).float()
    d = (x - y).abs()
    return float(d.max()), int((d > 0).sum())


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
    if len(recs) > len(rows):
        recs = recs[len(recs) - len(rows) :]
    elif len(rows) != len(recs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    by = collections.defaultdict(list)
    for rec, row in zip(recs, rows):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        by[(rec["case"], rec["variant"])].append(rec["ns"])
    out = []
    for rec in recs:
        if "ns" not in rec:
            continue
        v = rec["variant"]
        bkey = "base_wm" if v.endswith("_wm") else ("base_w" if v.endswith("_w") else "base")
        if "@" in v:
            bkey = "base@" + v.split("@", 1)[1]
        b = by.get((rec["case"], bkey))
        rel = f"{rec['ns'] / statistics.median(b):6.3f}x" if b else "   -   "
        md = None if rec.get("ablated") else _maxdiff(rec["case"], rec["variant"])
        mds = "-" if md is None else f"maxdiff {md[0]:.4g} ({md[1]} differ)"
        print(
            f"{rec['case']:<22} {rec['variant']:<14} r{rec.get('rep', 0)} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f} "
            f"mae {rec['mae']:.5f}  {mds}  {rec.get('blocking')}"
        )
        out.append(dict(rec, tag=args.tag, csv=str(args.csv), maxdiff_vs_base=md))
    print("medians:")
    for (case, var), v in by.items():
        print(f"  {case:<22} {var:<14} n={len(v)} median {statistics.median(v):>9.0f}  all {v}")
    if args.append:
        with open(args.append, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
