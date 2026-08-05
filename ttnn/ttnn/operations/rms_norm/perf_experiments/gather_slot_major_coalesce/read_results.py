# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join `manifests/<case>.jsonl` (launch ORDER + pcc/rel-RMS + knobs) to the profiler CSV's
DEVICE KERNEL DURATION [ns] rows and print the bake-off table.

    python3 read_results.py [ops_perf_results_*.csv] [case ...]

The manifests record one line per generic_op launch in launch order and the profiler CSV
lists device ops in the same order, so the join is POSITIONAL -- which is why the total line
count is ASSERTED against the CSV row count.  A mismatch means a stale manifest or an extra
device op, and a silently slid join would fabricate numbers.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MAN = HERE / "manifests"

# pytest parametrize order == launch order == CSV row order.
CASE_ORDER = [
    "focus",
    "br1",
    "br4",
    "br32",
    "gs4",
    "gs4_br1",
    "gs9",
    "gs28",
    "gs32",
    "gs32_multi",
    "gs16_multi",
    "ilv_gw8",
]


def main(argv):
    csv_path = next((a for a in argv if a.endswith(".csv")), None)
    if csv_path is None:
        cands = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))
        assert cands, "no profiler CSV found"
        csv_path = cands[-1]
    only = [a for a in argv if not a.endswith(".csv") and not a.startswith("--")]
    out_name = "measured.json" if not only else f"measured_{'_'.join(only)}.json"

    cases = [c for c in CASE_ORDER if (MAN / f"{c}.jsonl").exists() and (not only or c in only)]
    man = []
    for c in cases:
        man += [json.loads(l) for l in (MAN / f"{c}.jsonl").read_text().splitlines() if l.strip()]

    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()]
    print(f"csv={csv_path}\ncases={cases}  csv_rows={len(rows)}  manifest_lines={len(man)}")
    assert len(rows) == len(man), (
        f"positional join would slide: {len(rows)} CSV rows vs {len(man)} manifest lines. "
        f"Re-run the profiled pytest for exactly these cases."
    )
    for r, row in zip(man, rows):
        r["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))

    by_case = {}
    for r in man:
        by_case.setdefault(r["case"], []).append(r)

    hdr = (
        f"{'case':<11} {'variant':<14} {'BR':>3} {'GS':>3} {'nb':>3} {'txn':>4} {'txnB':>6} "
        f"{'bootB':>7} {'rootB':>7} {'ns':>8} {'vs base':>8} {'pcc':>9} {'relrms':>9} {'bitex':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for case, recs in by_case.items():
        base = next((r["ns"] for r in recs if r["name"] == "rm_f2"), None)
        for r in recs:
            spd = f"{base / r['ns']:.3f}x" if base and r["ns"] else "-"
            print(
                f"{case:<11} {r['name']:<14} {r['block_rows']:>3} {r['group_size']:>3} "
                f"{r['num_blocks']:>3} {r['gather_txns_per_member_round']:>4} "
                f"{r['gather_txn_bytes']:>6} {r['boot_zero_bytes']:>7} "
                f"{r.get('root_boot_zero_bytes', -1):>7} {r['ns']:>8} {spd:>8} "
                f"{r['pcc']:>9.6f} {r['rel_rms']:>9.2e} {str(r['bit_exact']):>6}"
            )
    (HERE / out_name).write_text(json.dumps(man, indent=1))
    print(f"\nwrote {HERE / out_name}")


if __name__ == "__main__":
    main(sys.argv[1:])
