# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join this experiment's launch log with the profiler CSV and print the menu.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/root_sum_dest_accumulate/report.py [csv]

`launches.jsonl` has one line per `ttnn.generic_op` launch, in launch order; the profiler's
ops_perf_results CSV has one row per device op, in the same order.  Columns printed:

  total_ns   DEVICE KERNEL DURATION [ns] for the whole launch (fold + drain + publish)
  fold_ns    total_ns minus the `floor` variant's total for the same geometry -- the
             ablation-subtracted cost of the accumulation mechanism alone
  x          fold_ns speedup vs `pack_l1_acc`, the op's CURRENT in-tree fold
  ns/row     fold_ns per tile-row (the unit the op's per-round zone divides into)
  L1d        extra fp32 gather pages this variant asks the descriptor for (+1cb = a new CB)
  pcc_out /  the op's soft gates, applied to x * rsqrt(sum/W + eps) with every other error
  rrms_out   source held exact, so they price THIS mechanism's precision only
  rrms_sum   relative RMS error of the raw group sum (the stage's own output)
"""

import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DUR = "DEVICE KERNEL DURATION [ns]"
BASELINE = "pack_l1_acc"


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
        f"{'variant':21s} {'total_ns':>9s} {'fold_ns':>8s} {'x':>6s} {'ns/row':>7s} {'L1d':>6s} "
        f"{'pcc_out':>10s} {'rrms_out':>9s} {'rrms_sum':>9s} {'gate':>5s}"
    )
    summary = []
    for k in keys:
        tag, g, rows_ = k
        grp = [r for r in recs if (r["tag"], r["group_size"], r["rows"]) == k]
        floor = next((r["total_ns"] for r in grp if r["variant"] == "floor"), None)
        base = next((r["total_ns"] - floor for r in grp if r["variant"] == BASELINE), None)
        print(f"\n=== {tag}  GROUP_SIZE={g}  rows={rows_}  (floor={floor} ns) ===")
        print(hdr)
        for r in grp:
            fold = (r["total_ns"] - floor) if floor is not None else None
            spd = (base / fold) if (base and fold) else None
            if r.get("ablation"):
                print(f"{r['variant']:21s} {r['total_ns']:9d} {'-':>8s} {'-':>6s} {'(ablation)':>7s}")
                continue
            l1d = f"{r['gather_l1_delta_pages']:+d}" + ("+1cb" if r.get("extra_cb_pages") else "")
            gate = "OK" if (r.get("pcc_gate_met") and r.get("relrms_gate_met")) else "FAIL"
            print(
                f"{r['variant']:21s} {r['total_ns']:9d} "
                f"{(fold if fold is not None else 0):8d} "
                f"{(f'{spd:.2f}x' if spd else '-'):>6s} "
                f"{(fold / rows_ if fold is not None else 0):7.1f} "
                f"{l1d:>6s} "
                f"{r['pcc_out']:10.6f} {r['rel_rms_out']:9.2e} {r['rel_rms_sum']:9.2e} {gate:>5s}"
            )
            summary.append((tag, g, rows_, r["variant"], fold, spd, gate))

    # Compact cross-geometry matrix for the variants that matter.
    print("\n=== speedup vs pack_l1_acc (fold_ns ratio), sweep tag ===")
    variants = [
        "pack_l1_acc_pairs",
        "dest_acc_wide",
        "dest_acc_wide_pad",
        "dest_pairs_tail_raw",
        "dest_acc_any",
    ]
    geoms = sorted({(g, r) for (t, g, r, *_rest) in summary if t == "sweep"})
    print(f"{'G,rows':>9s} " + " ".join(f"{v[:17]:>18s}" for v in variants))
    for g, r in geoms:
        cells = []
        for v in variants:
            hit = [s for s in summary if s[0] == "sweep" and s[1] == g and s[2] == r and s[3] == v]
            cells.append(f"{hit[0][4]:6d} {hit[0][5]:5.2f}x" if hit and hit[0][5] else f"{'--':>12s}")
        print(f"{f'{g},{r}':>9s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
