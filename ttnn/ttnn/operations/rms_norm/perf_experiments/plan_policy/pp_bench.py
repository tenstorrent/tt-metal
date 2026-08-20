# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The measurement half of `plan_policy`: what the plan knobs are actually worth.

Two sweeps, both driven entirely through the op's existing `_levers=` hook:

  * `sweep_groups`  — G (the combine group size) over every value the policy's own
    candidate enumeration accepts for that cell, plus G=1 (the row-parallel plan).
    The policy's pick is included as its own arm (`G:policy`) so the pick and the
    optimum are measured on the SAME dispatch order.
  * `sweep_chunk`   — the Regime-B W-chunk (`wt_block`), whose search objective is
    "coarsest divisor that fits".  Only meaningful where the cell solves to
    Regime B; a cap above Wt_core is a no-op arm and is skipped.

Every arm is PCC-gated on a warm-up dispatch before it is timed.
"""

from __future__ import annotations

from ttnn.operations.rms_norm.perf_experiments.plan_policy import pp_common as pp


def legal_groups(device, cell):
    """The G values the op's OWN policy enumerates for this cell (P1..P4 survivors)."""
    rows, _plan, pick = pp.candidate_table(device, cell)
    gs = sorted({r["g"] for r in rows})
    return gs, pick, rows


def _levers_for_g(g):
    if g == 1:
        return dict(w_split=0)
    return dict(w_group=g)


def sweep_groups(device, manifest, cell, groups=None, cap=None):
    """Time the policy pick and every legal G on one cell."""
    gs, pick, _rows = legal_groups(device, cell)
    if groups is not None:
        gs = [g for g in groups if g in gs]
    if cap and len(gs) > cap:  # keep the ends + a log-spaced interior
        keep = {gs[0], gs[-1], pick["G"]}
        step = max(1, len(gs) // cap)
        keep |= set(gs[::step])
        gs = sorted(keep)
    tensors = pp.make(device, cell)
    pp.run_arm(device, manifest, f"{cell}/G:policy", cell, None, tensors=tensors)
    for g in gs:
        pp.run_arm(device, manifest, f"{cell}/G:{g}", cell, _levers_for_g(g), tensors=tensors)
    return pick


def sweep_chunk(device, manifest, cell, caps=(1, 2, 4, 8, 16, 32, 64, 0)):
    """Time the Regime-B W-chunk cap (`wt_block`); 0 == the policy's own search."""
    _rows, _plan, pick = pp.candidate_table(device, cell)
    if pick["regime"] != "B":
        return pick  # Regime A never runs the chunk search
    tensors = pp.make(device, cell)
    wt_core = pick["Wt_core"]
    for c in caps:
        if c and (c > wt_core):
            continue
        levers = None if c == 0 else dict(wt_block=c)
        pp.run_arm(device, manifest, f"{cell}/WT:{c or 'policy'}", cell, levers, tensors=tensors)
    return pick


def sweep_knob(device, manifest, cell, knob, values, extra=None):
    """Time one plan knob over `values` on one cell; `None` value = the policy's pick.

    `extra` pins other levers for the whole sweep (e.g. a G, so a block_ht sweep
    is measured on the plan the policy actually ships).
    """
    tensors = pp.make(device, cell)
    for v in values:
        levers = dict(extra or {})
        tag = "policy"
        if v is not None:
            levers[knob] = v
            tag = str(v)
        pp.run_arm(device, manifest, f"{cell}/{knob}:{tag}", cell, levers or None, tensors=tensors)
