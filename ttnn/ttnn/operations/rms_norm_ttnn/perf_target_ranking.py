# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rank `feature_spec.LOOSE_CASES`' `perf` group by measured/achievable.

No `feature_spec.LOOSE_CASES` entry for this op carries an `attention` note, so
`eval/prompts/perf_refinement_prompt.txt` step 1 falls through to its second
branch: *"rank the `perf` group by measured device-ns divided by each case's own
`achievable_ns` and take the worst — absolute duration cannot compare shapes at
different placements."*  This script computes exactly that ranking, so the
verifier's queue and the trailing perf pass point at the same shape.

Inputs are the sidecars `eval/eval_test_runner.sh` already writes for a golden
run (it exports the three profiler env vars by default, so `device_kernel_ns` is
captured per test):

    <results_dir>/test_results.json   status + metrics, incl. device_kernel_ns
    <results_dir>/test_extras.json    the loose case's `extras` dict
    <results_dir>/test_groups.json    the `case_group` label

The reference is CLOCK-SCALED before the ratio, exactly as the feature spec's
header specifies:

    scaled_reference_ns = achievable_ns * reference_aiclk_mhz / actual_aiclk_mhz
    ceiling_ns          = scaled_reference_ns / minimum_expected_speedup   (when present)
    ratio               = measured_ns / ceiling_ns        (> 1.0 == MISSES)

`--aiclk` supplies the measured clock from device evidence.  Omitted, it
defaults to the reference clock (factor 1.0) and says so, because assuming a
nominal board default is precisely what the spec header forbids.

Usage:
    python3 -m ttnn.ttnn.operations.rms_norm_ttnn.perf_target_ranking <results_dir> [--aiclk MHZ]
    python3 ttnn/ttnn/operations/rms_norm_ttnn/perf_target_ranking.py <results_dir> [--aiclk MHZ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def _metric(record, name):
    """device_kernel_ns lives either flat on the record or under `metrics`."""
    if not isinstance(record, dict):
        return None
    if record.get(name) is not None:
        return record[name]
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(name)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument(
        "--aiclk",
        type=float,
        default=None,
        help="measured AICLK in MHz, from device/profiler evidence (not a board nominal)",
    )
    ap.add_argument("--group", default="perf", help="loose-case group to rank (default: perf)")
    args = ap.parse_args(argv)

    root = Path(args.results_dir)
    results = _load(root / "test_results.json")
    extras = _load(root / "test_extras.json")
    groups = _load(root / "test_groups.json")

    # test_results.json is a LIST of per-test records (eval_test_runner.sh's
    # shape), or a dict wrapping one. Index it by nodeid either way.
    if isinstance(results, dict):
        results = results.get("tests", results)
    if isinstance(results, list):
        results = {r.get("nodeid") or r.get("test_name"): r for r in results}

    rows = []
    for nodeid, extra in extras.items():
        if not isinstance(extra, dict) or extra.get("achievable_ns") is None:
            continue
        if args.group and groups.get(nodeid) not in (None, args.group):
            continue
        record = results.get(nodeid) or {}
        status = record.get("status") if isinstance(record, dict) else None
        measured = _metric(record, "device_kernel_ns")
        cores = _metric(record, "device_num_cores")

        achievable = float(extra["achievable_ns"])
        ref_clk = float(extra.get("reference_aiclk_mhz") or 0) or None
        actual_clk = args.aiclk
        if ref_clk and actual_clk:
            scaled = achievable * ref_clk / actual_clk
            clock_note = f"x{ref_clk / actual_clk:.4f}"
        else:
            scaled = achievable
            clock_note = "x1.0000 (assumed: --aiclk not supplied)"
        speedup = float(extra.get("minimum_expected_speedup") or 1.0)
        ceiling = scaled / speedup

        ratio = (float(measured) / ceiling) if measured else None
        rows.append(
            dict(
                nodeid=nodeid,
                status=status,
                measured=measured,
                cores=cores,
                achievable=achievable,
                scaled=scaled,
                speedup=speedup,
                ceiling=ceiling,
                ratio=ratio,
                clock_note=clock_note,
            )
        )

    if not rows:
        print(f"no '{args.group}'-group cases with achievable_ns found under {root}", file=sys.stderr)
        return 1

    # Worst first: a missing measurement sorts to the top, because an unmeasured
    # target is not evidence of meeting it.
    rows.sort(key=lambda r: (r["ratio"] is not None, -(r["ratio"] or 0.0)))

    print(f"# perf-group ranking (worst measured/ceiling first) — {len(rows)} cases")
    print(f"# clock scale: {rows[0]['clock_note']}")
    print(f"{'ratio':>7}  {'measured':>10}  {'ceiling':>10}  {'cores':>5}  {'status':>7}  case")
    for r in rows:
        ratio = f"{r['ratio']:.3f}" if r["ratio"] is not None else "  n/a"
        measured = f"{r['measured']:.0f}" if r["measured"] else "n/a"
        flag = "MISS" if (r["ratio"] or 0) > 1.0 else "ok"
        # The parametrize id carries the whole config; drop the file/function prefix.
        case = r["nodeid"].split("[", 1)[-1].rstrip("]")
        print(
            f"{ratio:>7}  {measured:>10}  {r['ceiling']:>10.0f}  {str(r['cores']):>5}  {str(r['status']):>7}  {flag} {case}"
        )

    misses = [r for r in rows if (r["ratio"] or 0) > 1.0]
    unmeasured = [r for r in rows if r["ratio"] is None]
    print(
        f"\n# {len(misses)} of {len(rows)} miss their ceiling; "
        f"{len(unmeasured)} unmeasured (profiler off, or the case did not run)"
    )
    if misses:
        print(f"# worst: {misses[0]['nodeid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
