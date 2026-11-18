# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pairs a profiled test_hbf_perf session with its records (no device, no ttnn import).

    python3 hbf_report.py <ops_perf_results csv> <HBF_TAG> [--zones z1,z2,...] [--device-log <csv>]

GenericOp rows of the CSV are the conv2d_3x3 dispatches in call order; the last len(rows) perf records with the tag
are the real pass. Zones come from the device log (default: profile_log_device.csv next to the CSV; pass a copy, the
shared one is overwritten by sibling runs): per-core sum over the dispatch, max and p50 over cores, per RISC.
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

DEFAULT_ZONES = "compute_matmul_walk,compute_matmul_stream,compute_bias_add,compute_bias_wait,compute_tilize"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("tag")
    ap.add_argument("--zones", default=DEFAULT_ZONES)
    ap.add_argument("--device-log", default=None)
    args = ap.parse_args()
    csv_path = Path(args.csv)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[header.index("OP CODE")].startswith("GenericOp")]
    col = header.index("DEVICE KERNEL DURATION [ns]")
    recs = [json.loads(line) for line in open(HERE / "results.jsonl")]
    recs = [r for r in recs if r.get("kind") == "perf" and r.get("tag") == args.tag]
    if len(recs) >= len(rows):
        recs = recs[len(recs) - len(rows) :]
    else:
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} records")
    dev_log = Path(args.device_log) if args.device_log else csv_path.parent / "profile_log_device.csv"
    freq, runs = zone_report.load(dev_log) if dev_log.exists() else (None, {})
    run_ids = sorted(rid for rid, ev in runs.items() if len({c for c, *_ in ev}) >= 8)
    zones = [z for z in args.zones.split(",") if z]
    out = []
    for i, (rec, row) in enumerate(zip(recs, rows)):
        rec["ns"] = int(float(row[col]))
        z = {}
        if len(run_ids) == len(rows):
            per_zone, _, _ = zone_report.summarize(runs[run_ids[i]], freq)
            for (risc, zone), v in per_zone.items():
                if zone in zones:
                    z[f"{zone}@{risc}"] = (round(max(v), 2), round(statistics.median(v), 2))
        rec["zones"] = z
        out.append(rec)
        print(
            f"{rec['case']:28s} {rec['variant']:9s} {rec['ns'] / 1000:8.2f} us  pcc {rec['pcc']:.6f} mae {rec['mae']:.5f}"
            f"  bm{rec.get('bm')} bk{rec.get('bk')} nkb{rec.get('nkb')} nb{rec.get('nb')} c{rec.get('center')}"
        )
        for k in sorted(z):
            print(f"      {k:40s} max {z[k][0]:7.2f}  p50 {z[k][1]:7.2f}")
    with open(HERE / "results_perf.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
