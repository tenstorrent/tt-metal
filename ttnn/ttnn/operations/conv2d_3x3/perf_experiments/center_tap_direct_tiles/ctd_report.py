# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pairs a profiled test_ctd_perf session with its records (no device, no ttnn import).

    python3 ctd_report.py <ops_perf_results csv> <CTD_TAG> [--zones z1,z2,...]

GenericOp rows of the CSV are the conv2d_3x3 dispatches in call order; the last len(rows) perf records with the tag
are the real pass (a precompile collect pass may log every body once before it). DEVICE KERNEL DURATION [ns] is
column 20. Zones (when the run kept them) come from the profile_log_device.csv next to the CSV: per-core sum over the
dispatch, max and p50 over cores, per RISC (TRISC zones are summed over the 3 TRISCs, so divide by 3 for wall time).
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

DEFAULT_ZONES = "compute_untilize,compute_tilize,compute_matmul_walk,reader_fill,reader_gather,reader_center"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("tag")
    ap.add_argument("--zones", default=DEFAULT_ZONES)
    args = ap.parse_args()
    csv_path = Path(args.csv)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[header.index("OP CODE")].startswith("GenericOp")]
    assert header[20 - 1].startswith("DEVICE KERNEL DURATION"), header[19]
    recs = [json.loads(line) for line in open(HERE / "results.jsonl")]
    recs = [r for r in recs if r.get("kind") == "perf" and r.get("tag") == args.tag]
    if len(recs) >= len(rows):
        recs = recs[len(recs) - len(rows) :]
    else:
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(recs)} records")
    dev_log = csv_path.parent / "profile_log_device.csv"
    freq, runs = zone_report.load(dev_log) if dev_log.exists() else (None, {})
    run_ids = sorted(rid for rid, ev in runs.items() if len({c for c, *_ in ev}) >= 8)
    zones = [z for z in args.zones.split(",") if z]
    out = []
    for i, (rec, row) in enumerate(zip(recs, rows)):
        rec["ns"] = int(float(row[19]))
        z = {}
        if i < len(run_ids):
            per_zone, _, _ = zone_report.summarize(runs[run_ids[i]], freq)
            for (risc, zone), v in per_zone.items():
                if zone in zones:
                    z[f"{zone}@{risc}"] = (round(max(v), 1), round(statistics.median(v), 1))
        rec["zones"] = z
        out.append(rec)
    for rec in out:
        blk = f"bm {rec.get('bm')} bk {rec.get('bk')} nkb {rec.get('nkb')} ring {rec.get('ring')} c {rec.get('center')}"
        print(f"{rec['case']:<26} {rec['variant']:<8} {rec['ns']:>9} ns  pcc {rec['pcc']:.6f}  {blk}")
        for k, (mx, p50) in sorted(rec["zones"].items()):
            print(f"      {k:<40} max {mx:8.1f}  p50 {p50:8.1f}")
    with open(HERE / "results_perf.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(dict(rec, csv=str(csv_path))) + "\n")


if __name__ == "__main__":
    main()
