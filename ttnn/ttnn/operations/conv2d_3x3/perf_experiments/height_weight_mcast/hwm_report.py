# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pairs a profiled test_hwm_perf session with its records (no device, no ttnn import).

    python3 hk_report.py <ops_perf_results csv> <HK_TAG> [--zones z1,z2,...] [--timeline x,y[;x,y]]

GenericOp rows of the CSV are the conv2d_3x3 dispatches in call order; the last len(rows) perf records with the tag
are the real pass. DEVICE KERNEL DURATION [ns] is column 20. Zones come from the profile_log_device.csv next to the
CSV: per-core sum over the dispatch, max and p50 over cores, per RISC. --timeline prints the zone start/end (us from
the dispatch's first marker on any core) of the given cores, every RISC.
"""

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import zone_report  # noqa: E402

DEFAULT_ZONES = "compute_matmul_walk,writer_bias,writer_weight_fetch,writer_weight_reserve,writer_w_read,writer_w_ackwait,writer_w_send,writer_w_wait,reader_fill,writer_fill,server_read,server_wait_read,server_wait_ack,server_send,server_issue,server_flush,server_wait_chain"


def timeline(events, freq, cores):
    t0 = min(e[2] for e in events)
    for core in cores:
        ev = [e for e in events if e[0] == core]
        stack = collections.defaultdict(list)
        rows = []
        for c, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
            if kind == "ZONE_START":
                stack[(risc, zone)].append(t)
            elif kind == "ZONE_END" and stack[(risc, zone)]:
                s = stack[(risc, zone)].pop()
                rows.append(((s - t0) / freq, (t - t0) / freq, risc, zone))
        print(f"--- core {core}")
        for s, e, risc, zone in sorted(rows):
            print(f"  {s:7.2f} {e:7.2f} {e - s:6.2f} {risc:8s} {zone}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("tag")
    ap.add_argument("--zones", default=DEFAULT_ZONES)
    ap.add_argument("--timeline", default="")
    ap.add_argument("--run-index", type=int, default=None, help="only this dispatch (index into the tagged records)")
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
        if args.timeline and (args.run_index is None or args.run_index == i) and i < len(run_ids):
            cores = [tuple(int(v) for v in c.split(",")) for c in args.timeline.split(";")]
            print(f"=== {rec['case']} {rec['variant']}")
            timeline(runs[run_ids[i]], freq, cores)
    for rec in out:
        blk = f"bm {rec.get('bm')} bk {rec.get('bk')} nb {rec.get('nb')} nkb {rec.get('nkb')} wd {rec.get('wd')} ub {rec.get('ub')} ring {rec.get('ring')} c {rec.get('center')}"
        print(f"{rec['case']:<40} {rec['variant']:<10} {rec['ns']:>9} ns  pcc {rec['pcc']:.6f}  {blk}")
        for k, (mx, p50) in sorted(rec["zones"].items()):
            print(f"      {k:<40} max {mx:8.1f}  p50 {p50:8.1f}")
    with open(HERE / "results_perf.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps(dict(rec, csv=str(csv_path))) + "\n")


if __name__ == "__main__":
    main()
