# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join `manifests/<case>.jsonl` (launch ORDER + pcc/rel-RMS + knobs, one file per
case, truncated by the test that owns it) to the profiler CSV's DEVICE KERNEL
DURATION [ns] rows, and print the bake-off table.

    python3 read_results.py [ops_perf_results_*.csv] [case ...]

The manifests record one line per generic_op launch in launch order and the profiler
CSV lists device ops in the same order, so the join is positional -- which is why the
total line count is ASSERTED against the CSV row count: a mismatch means a stale
manifest or an extra device op, and a silently slid join would fabricate numbers.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent
MAN = HERE / "manifests"

# `generated/profiler` is SHARED with the sibling benches running in this same clone, so
# every run of this bench sets TT_METAL_PROFILER_DIR to a private tree (tools/tracy/
# common.py honours it) and the reports are looked for there first.  A collision was
# caught here for real: a sibling's 28-row report was the "newest" one and would have
# been joined positionally to this bench's 18 launches.
PROFILER_DIRS = [os.environ.get("RMS_ROT_PROFILER_DIR", "/tmp/rms_root_rotation_profiler"), "generated/profiler"]

# pytest parametrize order == launch order == CSV row order.
CASE_ORDER = [
    "smoke",
    "focus",
    "nb1",
    "nb2",
    "nb8",
    "nb16",
    "gs4",
    "gs4_nb8",
    "gs32_nb1",
    "gs28_nb1",
    "gs9_nb1",
    "gs32_multi",
    "gs16_multi",
    "ilv_gw8",
]


def _rows(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()]


def _matches(rows, man):
    """Does this CSV's row sequence belong to THIS manifest set?  Row count AND the
    per-row input width must agree.  `generated/` is shared with the sibling benches in
    this clone, so the newest CSV is not reliably ours -- a foreign report with the same
    row count would otherwise be joined positionally and fabricate every number."""
    if len(rows) != len(man):
        return False

    def dims(row):
        # (rows, width) of input 0 -- the CSV spells the logical shape as
        # INPUT_0_{W,Z,Y,X}_PAD[LOGICAL], so Y is the row count and X the width.
        # The cells read "8192[8192]" (padded[logical]); take the leading integer.
        try:
            return tuple(int(str(row[f"INPUT_0_{ax}_PAD[LOGICAL]"]).split("[")[0]) for ax in ("Y", "X"))
        except (KeyError, TypeError, ValueError):
            return None

    return all(dims(r) == (m["h"], m["w"]) for r, m in zip(rows, man))


def main(argv):
    only = [a for a in argv if not a.endswith(".csv") and not a.startswith("--")]
    cases = [c for c in CASE_ORDER if (MAN / f"{c}.jsonl").exists() and (not only or c in only)]
    man = []
    for c in cases:
        man += [json.loads(l) for l in (MAN / f"{c}.jsonl").read_text().splitlines() if l.strip()]

    explicit = next((a for a in argv if a.endswith(".csv")), None)
    if explicit:
        csv_path, rows = explicit, _rows(explicit)
    else:
        cands = []
        for d in PROFILER_DIRS:
            cands += glob.glob(f"{d}/reports/*/ops_perf_results_*.csv")
        cands.sort(key=lambda p: -Path(p).stat().st_mtime)
        assert cands, "no profiler CSV found"
        hit = next(((p, _rows(p)) for p in cands if _matches(_rows(p), man)), None)
        assert hit is not None, (
            f"no ops_perf_results CSV matches these {len(man)} launches (row count + input H x W). "
            f"Newest candidates: {cands[:3]}.  Re-run the profiled pytest for exactly these cases."
        )
        csv_path, rows = hit
    print(f"csv={csv_path}\ncases={cases}  csv_rows={len(rows)}  manifest_lines={len(man)}")
    assert _matches(rows, man), "CSV does not match the manifest set"
    for r, row in zip(man, rows):
        r["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))

    by_case = {}
    for r in man:
        by_case.setdefault(r["case"], []).append(r)

    hdr = (
        f"{'case':<11} {'variant':<20} {'BR':>3} {'GS':>3} {'nb':>3} {'rc':>3} "
        f"{'ns':>8} {'vs fixed':>9} {'+L1':>5} {'pcc':>9} {'relrms':>9} {'bitex':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for case, recs in by_case.items():
        base = next((r["ns"] for r in recs if r["variant"] == 0), None)
        for r in recs:
            spd = f"{base / r['ns']:.3f}x" if base and r["ns"] else "-"
            tag = " (ABL)" if r.get("ablation") else ""
            print(
                f"{case:<11} {r['name'] + tag:<20} {r['block_rows']:>3} {r['group_size']:>3} "
                f"{r['num_blocks']:>3} {r['root_cores']:>3} {r['ns']:>8} {spd:>9} "
                f"{r['extra_l1_bytes']:>5} {r['pcc']:>9.6f} {r['rel_rms']:>9.2e} {str(r['bit_exact']):>6}"
            )
    # MERGE, don't overwrite: the sweep runs in several pytest invocations (the
    # --profile wrapper loses the quoting of a multi-word -k, so each batch is one
    # single-token filter and gets its own CSV), and measured.json is the union.
    out = HERE / "measured.json"
    prev = json.loads(out.read_text()) if out.exists() else []
    keep = {(r["case"], r["variant"]): r for r in prev if r["case"] not in {m["case"] for m in man}}
    merged = list(keep.values()) + man
    merged.sort(key=lambda r: (CASE_ORDER.index(r["case"]) if r["case"] in CASE_ORDER else 99, r["variant"]))
    out.write_text(json.dumps(merged, indent=1))
    print(f"\nwrote {HERE / 'measured.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
