# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join this experiment's launch log with the profiler CSV and print the menu.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/root_chain_dest_fuse/report.py [csv]

`launches.jsonl` has one line per `ttnn.generic_op` launch, in launch order; the
profiler's ops_perf_results CSV has one row per device op, in the same order.  Columns:

  total_ns    DEVICE KERNEL DURATION [ns] for the whole launch
  stage_ns    total_ns minus the `floor` variant's total for the same geometry -- the
              ablation-subtracted cost of the (root_sum + root_finalize) stage PAIR
  x           stage_ns speedup vs `baseline` (the op's HEAD approach)
  pcc_out /   the op's soft gates, applied to x * rsqrt(sum/W + eps) with every other
  rrms_out    error source held exact, so they price THIS mechanism's precision only
  rrms_stat   relative RMS error of the finalized stat itself (the stage pair's output)
"""

import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DUR = "DEVICE KERNEL DURATION [ns]"


def load_csv(path=None):
    if path is None:
        cands = sorted(
            glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=lambda p: Path(p).stat().st_mtime
        )
        if not cands:
            sys.exit("no ops_perf_results_*.csv under generated/profiler/reports/")
        path = cands[-1]
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(f"# csv: {path}  ({len(rows)} device ops)")
    return rows


def main():
    launches = [json.loads(l) for l in (HERE / "launches.jsonl").read_text().splitlines() if l.strip()]
    rows = load_csv(sys.argv[1] if len(sys.argv) > 1 else None)
    if len(rows) != len(launches):
        keep = [r for r in rows if "generic" in (r.get("OP CODE") or "").lower()]
        print(f"# {len(rows)} csv rows vs {len(launches)} launches -> filtering to OP CODE ~ generic: {len(keep)}")
        rows = keep
    if len(rows) != len(launches):
        sys.exit(f"cannot align: {len(rows)} csv rows vs {len(launches)} launches")

    recs = []
    for lau, row in zip(launches, rows):
        r = dict(lau)
        r["total_ns"] = int(float(row[DUR]))
        recs.append(r)

    keys = []
    for r in recs:
        k = (r["tag"], r["group_size"], r["rows"])
        if k not in keys:
            keys.append(k)

    hdr = (
        f"{'variant':20s} {'total_ns':>9s} {'stage_ns':>9s} {'x':>6s} "
        f"{'pcc_out':>10s} {'rrms_out':>9s} {'rrms_stat':>10s} {'gate':>5s}"
    )
    for k in keys:
        tag, g, rows_ = k
        grp = [r for r in recs if (r["tag"], r["group_size"], r["rows"]) == k]
        floor = next((r["total_ns"] for r in grp if r["variant"] == "floor"), None)
        base = None
        print(f"\n=== {tag}  GROUP_SIZE={g}  rows={rows_}  (floor={floor} ns) ===")
        print(hdr)
        for r in grp:
            stage = (r["total_ns"] - floor) if floor is not None else None
            if r["variant"] == "baseline":
                base = stage
            spd = (base / stage) if (base and stage) else None
            if r.get("ablation"):
                print(f"{r['variant']:20s} {r['total_ns']:9d} {'-':>9s} {'-':>6s} {'(ablation)':>10s}")
                continue
            gate = "OK" if (r.get("pcc_gate_met") and r.get("relrms_gate_met")) else "FAIL"
            print(
                f"{r['variant']:20s} {r['total_ns']:9d} "
                f"{(stage if stage is not None else 0):9d} "
                f"{(f'{spd:.2f}x' if spd else '-'):>6s} "
                f"{r['pcc_out']:10.6f} {r['rel_rms_out']:9.2e} {r['rel_rms_stat']:10.2e} {gate:>5s}"
            )


if __name__ == "__main__":
    main()
