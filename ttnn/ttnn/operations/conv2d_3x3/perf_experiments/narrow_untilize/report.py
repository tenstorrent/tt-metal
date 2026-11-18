# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a profiled run of test_narrow_untilize_op.py with its run log (no device needed).

    python3 report.py <ops_perf_results csv> [runs/run_<pid>.jsonl] [--zones zone1,zone2]

GenericOp rows of the CSV are the op dispatches in call order; the run log's last len(rows) records are the real pass
(a precompile collect pass logs every body once more, before it). Zones come from the profile_log_device.csv next to
the CSV: per-core sum over the run, p50 over cores, per RISC.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import zone_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--zones", default="compute_untilize,reader_fill,reader_band_wait,reader_gather")
    args = ap.parse_args()
    csv_path = Path(args.csv)
    log_path = Path(args.log) if args.log else max((HERE / "runs").glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["OP CODE"].startswith("GenericOp")]
    recs = [json.loads(line) for line in open(log_path)]
    if len(recs) >= len(rows):
        recs = recs[len(recs) - len(rows) :]
    else:
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} logged dispatches")
    dev_log = csv_path.parent / "profile_log_device.csv"
    freq, runs = zone_report.load(dev_log) if dev_log.exists() else (None, {})
    run_ids = sorted(runs)
    zones = [z for z in args.zones.split(",") if z]
    base = {}
    out = []
    for i, (rec, row) in enumerate(zip(recs, rows)):
        rec["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
        rid = int(row.get("GLOBAL CALL COUNT", 0) or 0)
        zinfo = {}
        if runs:
            key = rid if rid in runs else (run_ids[i] if i < len(run_ids) else None)
            if key is not None:
                per_zone, _, _ = zone_report.summarize(runs[key], freq)
                for (risc, zone), v in per_zone.items():
                    if zone in zones:
                        zinfo[f"{zone}@{risc}"] = round(statistics.median(v), 2)
        rec["zones"] = zinfo
        if rec["variant"] == "base":
            base.setdefault(rec["case"], []).append(rec["ns"])
        out.append(rec)
    for rec in out:
        b = base.get(rec["case"])
        rel = f"{statistics.median(b) / rec['ns']:6.3f}x" if b else "   -   "
        z = "  ".join(f"{k}={v}" for k, v in sorted(rec["zones"].items()))
        print(
            f"{rec['case']:<26} {rec['order']} {rec['variant']:<9} {rec['ns']:>9} ns  {rel}  pcc {rec['pcc']}  "
            f"eq {rec['equal_first']}  {z}"
        )
    with open(HERE / "results_op.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(dict(rec, csv=str(csv_path))) + "\n")


if __name__ == "__main__":
    main()
