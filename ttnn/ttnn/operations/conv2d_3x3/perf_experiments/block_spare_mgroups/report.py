# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run's ops_perf_results CSV (GenericOp rows, in call order) with this harness's run log.

    python3 report.py <ops_perf_results.csv> <runs/run_*.jsonl> [--append results.jsonl --tag TAG]

Bit-exactness and max abs diff are judged against the `baseline` variant of the same case (its saved output,
runs/outs/<case>-baseline.pt, from this or an earlier run of the same seeded inputs).
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def _maxdiff(case, variant):
    try:
        import torch
    except ImportError:
        return None
    a = HERE / "runs" / "outs" / f"{case.replace(':', '_')}-baseline.pt"
    b = HERE / "runs" / "outs" / f"{case.replace(':', '_')}-{variant}.pt"
    if not (a.exists() and b.exists()):
        return None
    x, y = torch.load(a).float(), torch.load(b).float()
    d = (x - y).abs()
    return float(d.max()), float((d > 0).float().mean())


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
        md = _maxdiff(rec["case"], rec["variant"]) if rec["variant"] != "baseline" else (0.0, 0.0)
        mds = "-" if md is None else f"maxdiff {md[0]:.4g} ({100 * md[1]:.3f}% differ)"
        print(
            f"{rec['case']:<22} {rec['variant']:<10} {rec['ns']:>9} ns {rel}  pcc {rec['pcc']:.6f} "
            f"(>= {rec['pcc_threshold']}) mae {rec['mae']:.5f} max_err {rec.get('max_err', 0):.4f}  {mds}  "
            f"{rec.get('blocking')}"
        )
        out.append(dict(rec, tag=args.tag, csv=str(args.csv), maxdiff_vs_base=md))
    if args.append:
        with open(args.append, "a") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
