# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join `manifests/<case>.jsonl` (launch ORDER + pcc + knobs, one file per case,
truncated by the test that owns it) to the profiler CSV's DEVICE KERNEL DURATION
[ns] rows, and print the bake-off table.

    python3 read_results.py [ops_perf_results_*.csv] [case ...]

The manifests record one line per generic_op launch in launch order and the
profiler CSV lists device ops in the same order, so the join is positional --
which is why the total line count is ASSERTED against the CSV row count: a
mismatch means a stale manifest or an extra device op, and a silently slid join
would fabricate numbers.
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
CASE_ORDER = ["focus", "nb1", "nb2", "nb8", "nb16", "gs32_nb1", "gs28_nb1", "gs32_multi", "gs16_multi", "ilv_gw8"]


def main(argv):
    csv_path = next((a for a in argv if a.endswith(".csv")), None)
    if csv_path is None:
        cands = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))
        assert cands, "no profiler CSV found"
        csv_path = cands[-1]
    only = [a for a in argv if not a.endswith(".csv") and not a.startswith("--")]

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

    hdr = f"{'case':<11} {'variant':<24} {'BR':>3} {'GS':>3} {'nb':>3} {'ns':>8} {'vs base':>8} {'+L1 B/core':>10} {'bitex':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for case, recs in by_case.items():
        base = next((r["ns"] for r in recs if r["variant"] == 0 and "@" not in r["name"]), None)
        for r in recs:
            spd = f"{base / r['ns']:.3f}x" if base and r["ns"] else "-"
            print(
                f"{case:<11} {r['name']:<24} {r['block_rows']:>3} {r['group_size']:>3} "
                f"{r['num_blocks']:>3} {r['ns']:>8} {spd:>8} {r['extra_l1_bytes']:>10} {str(r['bit_exact']):>6}"
            )
    (HERE / "measured.json").write_text(json.dumps(man, indent=1))
    print(f"\nwrote {HERE / 'measured.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
