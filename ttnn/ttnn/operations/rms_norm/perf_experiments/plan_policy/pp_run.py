# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Entry points the pytest driver shim calls.  Env-selected so the sweep can be
split across several device sessions (the JIT compile of each new G is paid
once and then cached on disk).

Env:
    PP_MODE   probe | groups | chunk
    PP_CELLS  comma list of pp_common.CELLS keys (default: all)
    PP_GS     comma list of G values to restrict the group sweep to
"""

from __future__ import annotations

import json
import os

from ttnn.operations.rms_norm.perf_experiments.plan_policy import pp_bench, pp_common as pp, pp_policy, pp_probe


def _cells():
    sel = os.environ.get("PP_CELLS")
    return [c for c in sel.split(",") if c] if sel else list(pp.CELLS)


def run(device):
    mode = os.environ.get("PP_MODE", "probe")
    if mode == "probe":
        pp_probe.run(device)
        return

    if mode == "plans":
        # Host-side only: what plan does a given lever set SOLVE to?  Used to read
        # the row-block geometry behind an `active_cores` / `w_group` arm.
        sets = json.loads(os.environ.get("PP_LEVERSETS", "[{}]"))
        for cell in _cells():
            print(f"\n=== {cell} ===")
            for lev in sets:
                pick = pp.candidate_table(device, cell, lev or None)[2]
                print(
                    f"  {json.dumps(lev):<40} G={pick['G']:<3} bht={pick['block_ht']:<3} "
                    f"nrb={pick['num_row_blocks']:<4} groups={pick['num_groups']}/{pick['groups_used']:<4} "
                    f"Wt_core={pick['Wt_core']:<4} wr={pick['wr']:<4} in={pick['in_depth']} out={pick['out_depth']} "
                    f"regime={pick['regime']} l1={pick['l1']}"
                )
        return

    if mode == "policy_plans":
        # Host-side: current policy vs the candidate, on every cell.
        guard = int(os.environ.get("PP_MIN_BLOCKS", "0"))
        variant = getattr(pp_policy, os.environ.get("PP_VARIANT", "depth_preserving_policy"))
        for cell in _cells():
            cur = pp.candidate_table(device, cell)[2]
            with variant(guard):
                new = pp.candidate_table(device, cell)[2]
            flag = "CHANGED" if (cur["G"], cur["block_ht"]) != (new["G"], new["block_ht"]) else "same"
            print(
                f"{cell:<20} {flag:<8} cur: G={cur['G']} bht={cur['block_ht']} nrb={cur['num_row_blocks']} "
                f"cores={min(cur['num_groups'], cur['num_row_blocks']) * cur['G']} in={cur['in_depth']} l1={cur['l1']}"
                f"   ->  new: G={new['G']} bht={new['block_ht']} nrb={new['num_row_blocks']} "
                f"cores={min(new['num_groups'], new['num_row_blocks']) * new['G']} in={new['in_depth']} l1={new['l1']}"
            )
        return

    if mode == "policy_bench":
        # The A/B: current policy vs the candidate, measured, repeated.
        guard = int(os.environ.get("PP_MIN_BLOCKS", "0"))
        reps = int(os.environ.get("PP_REPS", "3"))
        variant = getattr(pp_policy, os.environ.get("PP_VARIANT", "depth_preserving_policy"))
        manifest = []
        for cell in _cells():
            tensors = pp.make(device, cell)
            for r in range(reps):
                pp.run_arm(device, manifest, f"{cell}/cur#{r}", cell, None, tensors=tensors)
                with variant(guard):
                    pp.run_arm(device, manifest, f"{cell}/new#{r}", cell, None, tensors=tensors)
        path = pp.write_manifest(manifest, pp.ART / "manifest_policy_bench.json")
        print(f"\nPP: manifest -> {path} ({len(manifest)} arms)")
        for arm in manifest:
            print(f"  {arm['label']:<28} pcc={arm['pcc']}")
        return

    if mode == "sets":
        # An explicit list of lever sets, dispatched REP times in round-robin order,
        # so run-to-run drift shows up as a spread across repeats of the SAME arm
        # rather than as a fake difference between two arms.
        sets = json.loads(os.environ.get("PP_LEVERSETS", "[{}]"))
        reps = int(os.environ.get("PP_REPS", "3"))
        manifest = []
        for cell in _cells():
            tensors = pp.make(device, cell)
            for r in range(reps):
                for i, lev in enumerate(sets):
                    pp.run_arm(
                        device, manifest, f"{cell}/set{i}:{json.dumps(lev)}#{r}", cell, lev or None, tensors=tensors
                    )
        path = pp.write_manifest(manifest, pp.ART / "manifest_sets.json")
        print(f"\nPP: manifest -> {path} ({len(manifest)} arms)")
        for arm in manifest:
            print(f"  {arm['label']:<52} pcc={arm['pcc']}")
        return

    gs = os.environ.get("PP_GS")
    gs = [int(v) for v in gs.split(",")] if gs else None
    manifest = []
    for cell in _cells():
        if mode == "groups":
            pick = pp_bench.sweep_groups(device, manifest, cell, groups=gs)
        elif mode == "chunk":
            caps = os.environ.get("PP_CAPS")
            kw = dict(caps=[int(v) for v in caps.split(",")]) if caps else {}
            pick = pp_bench.sweep_chunk(device, manifest, cell, **kw)
        elif mode.startswith("knob:"):
            _, knob, vals = mode.split(":", 2)
            values = [None] + [int(v) for v in vals.split(",")]
            extra = json.loads(os.environ.get("PP_EXTRA", "{}"))
            pp_bench.sweep_knob(device, manifest, cell, knob, values, extra=extra)
            pick = pp.candidate_table(device, cell)[2]
        else:
            raise AssertionError(f"unknown PP_MODE {mode}")
        print(f"PP: {cell} policy pick {pick}")
    path = pp.write_manifest(manifest, pp.ART / f"manifest_{mode.replace(':','_')}.json")
    print(f"\nPP: manifest -> {path} ({len(manifest)} arms)")
    for arm in manifest:
        print(f"  {arm['label']:<30} pcc={arm['pcc']} levers={arm['levers']}")
    assert manifest, "bench dispatched nothing"
