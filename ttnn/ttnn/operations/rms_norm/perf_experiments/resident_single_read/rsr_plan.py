# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""resident_single_read lab: rms_norm with a THIRD blocking plan, Regime C.

THE IDEA
--------
The op has two plans, picked by one host predicate:

    A  RESIDENT-FUSED    ONE DRAM read of x.  Requires the ENTIRE minimal
                         resident working set to fit L1 - and that set includes
                         the full-width gamma CB, the full-width cb_normed and
                         the full-width output CB, not just x.
    B  STREAMING-MASKED  TWO DRAM reads of x (one for the sum of squares, one
                         for the scale pass) => 1.5x the DRAM bytes of A.

So a shape whose x slice WOULD fit resident still falls to B because the OTHER
four full-width CBs do not.  On the focus shape (1,1,32,7168) x resident is
224 x 2048 = 458,752 B against a 1,269,888 B budget - it fits four times over;
what does not fit is 5 x full width.

Regime C keeps ONLY x resident and chunks the scale pass:

    read x once (full per-core width) -> fused sum_of_squares -> finalize ->
    rms chain -> for each W-chunk: [gamma slice] x*(1/rms)*gamma -> out

gamma / cb_normed / cb_output_tiles are sized PER CHUNK, so the second DRAM read
of x disappears at a bounded L1 cost.  It is a MERGE of the two regimes: A's
input+reduce half, B's gamma+scale+write half.

WHAT THIS RENEGOTIATES
----------------------
The op asserts `regime == "A" implies WT_SCALE_BLOCK == Wt_core`, because
cb_gamma_tiles is never popped in Regime A so one pass-B call has to span every
gamma column from the CB front.  Regime C pops gamma per chunk and has the reader
re-push each slice (Regime B's protocol), which retires that constraint.  The
second constraint it renegotiates is the CB-WRAP INVARIANT: the resident
cb_input_tiles is read by the scale pass in chunks WITHOUT being popped between
them, so chunk c is not at the CB front.  That is addressed with
`TileOffset::Strided` (tile_id = c*ws + r*Wt_core + w) under caller-managed
(None, None) policies, with ONE pop of the whole BLOCK_HT x Wt_core window after
the last chunk - so every CB access is still a multiple of a fixed window from an
aligned fifo pointer.

KNOBS (0 / "" everywhere == behave exactly like the op)
    allow_c        0 -> Regime C is never selected  (THE HONEST BASELINE ARM)
    force_regime   "A" | "B" | "C" -> pin the plan (for counterfactual pricing)
    c_ws           force WT_SCALE_BLOCK in Regime C (must divide Wt_core)
    c_in_depth / c_out_depth / c_gamma_depth / c_normed_depth
                   force the per-CB depth in Regime C
    c_block_ht     force BLOCK_HT in Regime C
    c_fused_reduce 0 -> resident x but B's STREAMING reduce datapath (the
                   decomposition arm: separates "one read" from "fused reduce")
    no_zones       1 -> compile the kernels with the per-stage zones no-op'd

THE SHIPPABLE PREDICATE (a real property, not a shape list)
-----------------------------------------------------------
A 4-level ladder, each level tried in order and taken if its MINIMAL working set
fits the CB budget (all of it derived from `_rsr_cb_layout`, so it moves with the
dtype / DEST width / layout instead of being a hard-coded width):

    A   whole width resident (x + gamma + normed + out)   -> unchanged
    C1  x + gamma resident, normed/out chunked            -> NEW
    C2  x resident, gamma/normed/out chunked              -> NEW
    B   two DRAM reads of x                               -> unchanged fallback

plus Regime A's existing `maskless_w` requirement for C1/C2 (the fused reduce's
element-wise accumulator has no per-column mask position - see the domain notes).
Evaluated on TILE input with gamma, BLOCK_HT=1, depth 1, the ladder's boundaries
land at:

    bf16,      acc bf16 :  A <= Wt 154 | C1 <= 307 | C2 <= 613 | B above
    float32,   acc fp32 :  A <=    76  | C1 <= 152 | C2 <= 303 | B above
    bfloat8_b, acc bf16 :  A <=   237  | C1 <= 578 | C2 <= 1155 | B above

`assert_matches_op_plan()` is the honest-baseline gate: at the lab defaults with
allow_c=0 this module must reproduce `opd.blocking_plan()` field for field.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd
from ttnn.operations.rms_norm.rms_norm import validate

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = opd.TILE_DIM

LAB_DEFAULTS = dict(opd.LEVER_DEFAULTS)
LAB_DEFAULTS.update(
    {
        "allow_c": 1,
        "force_regime": "",
        "c_ws": 0,
        "c_in_depth": 0,
        "c_out_depth": 0,
        "c_gamma_depth": 0,
        "c_normed_depth": 0,
        "c_block_ht": 0,
        "c_fused_reduce": 1,
        # -1 = AUTO: keep Regime A's RESIDENT gamma (full per-core width, read
        # once per core, never popped) wherever it fits, else chunk gamma too
        # (Regime B's per-chunk gamma protocol).  0/1 force one form.
        "c_resident_gamma": -1,
        "no_zones": 1,
    }
)

REGIME_CODE = {"B": 0, "A": 1, "C": 2}


def _lever(levers, name):
    return LAB_DEFAULTS[name] if levers is None else levers.get(name, LAB_DEFAULTS[name])


# ---------------------------------------------------------------------------
# The CB set.  A and B mirror opd._cb_layout() EXACTLY (same list, same order);
# Regime C is the new branch.
# ---------------------------------------------------------------------------
def _rsr_cb_layout(
    *,
    regime: str,
    block_ht: int,
    in_depth: int,
    out_depth: int,
    rm_depth: int,
    gamma_depth: int,
    normed_depth: int,
    fused_reduce: int,
    resident_gamma: int,
    wr: int,
    ws: int,
    Wt_core: int,
    has_gamma: bool,
    gamma_is_row_major: bool,
    is_row_major: bool,
    tile_out: bool,
    W_partial: int,
    gamma_ingest_block: int,
    T_in: int,
    T_g: int,
    T_interm: int,
    T_acc: int,
    T_bf16: int,
):
    if regime != "C":
        return opd._cb_layout(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_depth,
            out_depth=out_depth,
            rm_depth=rm_depth,
            wr=wr,
            ws=ws,
            Wt_core=Wt_core,
            has_gamma=has_gamma,
            gamma_is_row_major=gamma_is_row_major,
            is_row_major=is_row_major,
            tile_out=tile_out,
            W_partial=W_partial,
            gamma_ingest_block=gamma_ingest_block,
            T_in=T_in,
            T_g=T_g,
            T_interm=T_interm,
            T_acc=T_acc,
            T_bf16=T_bf16,
        )

    # --- Regime C ---------------------------------------------------------
    # INPUT side: full per-core width (that IS the idea).
    # OUTPUT / gamma / normed side: one W-chunk.
    layout = [
        (opd.CB_INPUT_TILES, in_depth * block_ht * Wt_core, T_in, "in"),
        # Fused reduce writes cb_sumsq once per row-block (one generation, like
        # Regime A).  The decomposition arm accumulates across chunks through it
        # and needs the extra generation live, like Regime B.
        (opd.CB_SUMSQ, (1 if fused_reduce else 2) * block_ht, T_acc, "acc"),
        (opd.CB_RMS_RECIP, block_ht, T_acc, "acc"),
        # Regime C is maskless by construction (its predicate keeps Regime A's
        # `maskless_w`), so one scaler tile - the within-tile finalize's.
        (opd.CB_REDUCE_SCALER, 1, T_bf16, "bf16"),
    ]
    if fused_reduce:
        layout.append((opd.CB_SUMSQ_ACC, block_ht, T_acc, "acc"))
    else:
        layout.append((opd.CB_SQUARED, block_ht * ws, T_interm, "interm"))
    if has_gamma:
        # resident gamma costs the full width but is read ONCE per core.
        layout.append((opd.CB_GAMMA_TILES, Wt_core if resident_gamma else gamma_depth * ws, T_g, "gamma"))
        if gamma_is_row_major:
            layout.append((opd.CB_GAMMA_RM, gamma_ingest_block, T_g, "gamma"))
        layout.append((opd.CB_NORMED, normed_depth * block_ht * ws, T_interm, "interm"))
    layout.append((opd.CB_OUTPUT_TILES, (out_depth if tile_out else 1) * block_ht * ws, T_in, "out"))
    if is_row_major:
        # The RM sticks arrive full-width (one read), the untilize drains per chunk.
        layout.append((opd.CB_RM_IN, rm_depth * Wt_core, T_in, "in"))
        layout.append((opd.CB_RM_OUT, rm_depth * ws, T_in, "out"))
    return layout


@dataclass(frozen=True)
class LabPlan:
    Rt: int
    Wt: int
    Wt_core: int
    W_true: int
    W_partial: int
    num_rows: int
    is_row_major: bool
    has_gamma: bool
    gamma_is_row_major: bool
    tile_out: bool
    elem_size: int
    gamma_elem_size: int
    in_tile_bytes: int
    gamma_tile_bytes: int
    interm_dtype: object
    acc_dtype: object
    row_bytes: int
    gamma_row_bytes: int
    BLOCK_HT: int
    WT_REDUCE_BLOCK: int
    WT_SCALE_BLOCK: int
    DEST_BLOCK: int
    GAMMA_INGEST_BLOCK: int
    IN_BUF_DEPTH: int
    OUT_BUF_DEPTH: int
    RM_BUF_DEPTH: int
    GAMMA_DEPTH: int
    NORMED_DEPTH: int
    regime: str
    fused_reduce: int
    resident_gamma: int
    reduce_via_add: int
    num_row_blocks: int
    l1_cb_budget: int
    fits_a: bool
    fits_c: bool
    cb_layout: tuple

    def working_set_bytes(self) -> int:
        return sum(p * b for _, p, b, _ in self.cb_layout)


def lab_blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers=None) -> LabPlan:
    shape = list(input_tensor.shape)
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    Rt, Wt, W_true, W_partial, num_rows = opd.tile_geometry(shape, is_row_major)
    Wt_core = Wt

    has_gamma = gamma is not None
    gamma_is_row_major = bool(has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT)

    elem_size = opd._elem_size(input_tensor.dtype)
    gamma_elem_size = opd._elem_size(gamma.dtype) if has_gamma else elem_size

    interm_dtype = opd._interm_dtype(input_tensor.dtype)
    acc_dtype = opd._acc_dtype(compute_kernel_config, interm_dtype, bool(_lever(levers, "acc_narrow")))

    T_in = ttnn.tile_size(input_tensor.dtype)
    T_g = ttnn.tile_size(gamma.dtype) if has_gamma else T_in
    T_interm = ttnn.tile_size(interm_dtype)
    T_acc = ttnn.tile_size(acc_dtype)
    T_bf16 = ttnn.tile_size(ttnn.bfloat16)

    tile_out = not is_row_major
    l1_cb_budget = ttnn.get_max_worker_l1_unreserved_size() - opd.L1_RESERVED_BYTES

    dest_limit = opd._dest_limit(compute_kernel_config)
    forced_dest = _lever(levers, "dest_block")
    if forced_dest:
        dest_limit = min(dest_limit, forced_dest)

    common = dict(
        Wt_core=Wt_core,
        has_gamma=has_gamma,
        gamma_is_row_major=gamma_is_row_major,
        is_row_major=is_row_major,
        tile_out=tile_out,
        W_partial=W_partial,
        T_in=T_in,
        T_g=T_g,
        T_interm=T_interm,
        T_acc=T_acc,
        T_bf16=T_bf16,
    )

    gamma_cap_tiles = max(1, opd.GAMMA_STAGE_MAX_BYTES // T_g)
    fused_reduce = int(_lever(levers, "c_fused_reduce"))
    # -1 = AUTO (prefer resident gamma, fall back to chunked); 0/1 force.
    rg_lever = int(_lever(levers, "c_resident_gamma"))
    resident_gamma = 0 if rg_lever < 0 else rg_lever

    def ws_bytes(regime, block_ht, in_depth, out_depth, rm_depth, wr, wsc, gd=1, nd=1, rg=None):
        return sum(
            p * b
            for _, p, b, _ in _rsr_cb_layout(
                regime=regime,
                block_ht=block_ht,
                in_depth=in_depth,
                out_depth=out_depth,
                rm_depth=rm_depth,
                gamma_depth=gd,
                normed_depth=nd,
                fused_reduce=fused_reduce,
                resident_gamma=resident_gamma if rg is None else rg,
                wr=wr,
                ws=wsc,
                gamma_ingest_block=opd._largest_divisor_at_most(wsc, gamma_cap_tiles),
                **common,
            )
        )

    # --- regime selection ---------------------------------------------------
    #  (1) maskless_w - unchanged from the op: can the reduce see the padded
    #      columns without a mask?  Regime C inherits this requirement because it
    #      reaches the reduce through the same FUSED accumulate as Regime A,
    #      whose element-wise accumulator has no per-column mask position.
    #  (2) does the whole minimal resident set fit (Regime A)?
    #  (3) else: does X ALONE fit resident, with a one-tile scale chunk?
    maskless_w = is_row_major or (W_partial == 0)
    fits_a = ws_bytes("A", 1, 1, 1, 1, Wt_core, Wt_core) <= l1_cb_budget
    fits_c = ws_bytes("C", 1, 1, 1, 1, Wt_core, 1) <= l1_cb_budget
    allow_c = bool(_lever(levers, "allow_c"))

    forced = _lever(levers, "force_regime")
    if forced:
        regime = forced
    elif maskless_w and fits_a:
        regime = "A"
    elif maskless_w and fits_c and allow_c:
        regime = "C"
    else:
        regime = "B"

    if regime == "A" and not (maskless_w and fits_a):
        raise ValueError("force_regime=A: not expressible on this shape (mask or L1)")
    if regime == "C" and not (maskless_w and fits_c):
        raise ValueError("force_regime=C: not expressible on this shape (mask or L1)")

    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    core_cap = _lever(levers, "active_cores") or opd.ACTIVE_CORE_CAP
    if core_cap:
        grid_cores = min(grid_cores, core_cap)
    max_block_ht = max(1, opd._div_up(Rt, max(1, grid_cores)))
    max_block_ht = min(max_block_ht, dest_limit)

    block_ht = 1
    in_depth = out_depth = rm_depth = 1
    gamma_depth = normed_depth = 1

    if regime == "A":
        wr = wsc = Wt_core
    elif regime == "C":
        # DEPTHS FIRST, then the chunk.  Measured on the focus shape (Wt_core=224,
        # budget 620 pages): the coarsest chunk at depth 1 (ws=112) gives 32,016 ns,
        # while a MODERATE chunk whose gamma + output CBs are double-buffered gives
        # 30,2xx-30,5xx ns across ws = 28..56 - i.e. letting the reader run a chunk
        # ahead is worth more than a coarser chunk, and the curve is flat over a
        # wide band before per-chunk overhead takes over below ws=16 (35.4k at
        # ws=16 d1, 41k at ws=8 d1, 127k at ws=1).  So: pick the depths, then take
        # the coarsest DIVISOR of Wt_core (CB-wrap constraint) that still fits.
        f_in, f_out, f_g, f_n = (
            _lever(levers, "c_in_depth"),
            _lever(levers, "c_out_depth"),
            _lever(levers, "c_gamma_depth"),
            _lever(levers, "c_normed_depth"),
        )
        if f_in or f_out or f_g or f_n:
            depth_prefs = [(f_in or 1, f_out or 1, f_g or 1, f_n or 1)]
        elif _lever(levers, "double_buffer"):
            depth_prefs = [(1, 2, 2, 1), (1, 1, 1, 1)]
        else:
            depth_prefs = [(1, 1, 1, 1)]

        forced_ws = _lever(levers, "c_ws")
        # GAMMA RESIDENCY, preferred where it fits.  Measured: with more than one
        # row-block per core, chunked gamma is re-read once per row-block and costs
        # as many transactions as x itself (prefill (1,1,8192,7168): 955,835 ns
        # chunked vs 792,067 ns resident); with exactly one row-block it is a wash
        # (focus: 30,253 vs 30,387 ns).  So AUTO = try resident, then chunked.
        rg_prefs = [1, 0] if rg_lever < 0 else [rg_lever]
        wsc = 0
        for rg in rg_prefs:
            for di, do, dg, dn in depth_prefs:
                if forced_ws:
                    assert Wt_core % forced_ws == 0, f"c_ws={forced_ws} must divide Wt_core={Wt_core}"
                    cands = [forced_ws]
                else:
                    cands = [c for c in range(Wt_core, 0, -1) if Wt_core % c == 0]
                for cand in cands:
                    if ws_bytes("C", 1, di, do, di, Wt_core, cand, dg, dn, rg=rg) <= l1_cb_budget:
                        wsc = cand
                        resident_gamma = rg
                        in_depth = rm_depth = di
                        out_depth, gamma_depth, normed_depth = do, dg, dn
                        break
                if wsc:
                    break
            if wsc:
                break
        assert wsc, "Regime C: no (chunk, depth) assignment fits L1"
        wr = wsc  # Regime C has no separate reduce chunk (the reduce is one call)
    else:
        wr = wsc = 1
        if _lever(levers, "coarse_chunk"):
            forced_wt = _lever(levers, "wt_block")
            chunk_cap = min(Wt_core, forced_wt) if forced_wt else Wt_core
            for cand in range(chunk_cap, 0, -1):
                if Wt_core % cand != 0:
                    continue
                if ws_bytes("B", 1, 1, 1, 1, cand, cand) <= l1_cb_budget:
                    wr = wsc = cand
                    break

    def fits(bh, din, dout, drm, gd, nd):
        return ws_bytes(regime, bh, din, dout, drm, wr, wsc, gd, nd) <= l1_cb_budget

    if regime == "C":
        # (chunk + depths already solved together above)
        forced_bh = _lever(levers, "c_block_ht")
        if forced_bh:
            max_block_ht = min(max_block_ht, forced_bh)
    else:
        if _lever(levers, "double_buffer"):
            if fits(block_ht, 2, 2, 2, 1, 1):
                in_depth = out_depth = rm_depth = 2
            elif fits(block_ht, 2, 1, 2, 1, 1):
                in_depth = rm_depth = 2
            elif fits(block_ht, 1, 1, 2, 1, 1):
                rm_depth = 2

    forced_block_ht = _lever(levers, "block_ht")
    if forced_block_ht:
        max_block_ht = min(max_block_ht, forced_block_ht)

    while block_ht < max_block_ht and fits(block_ht + 1, in_depth, out_depth, rm_depth, gamma_depth, normed_depth):
        block_ht += 1

    if regime != "C" and _lever(levers, "double_buffer"):
        while in_depth < 4 and fits(block_ht, in_depth + 1, out_depth, rm_depth, gamma_depth, normed_depth):
            in_depth += 1

    assert Wt_core % wr == 0 and Wt_core % wsc == 0, "W-chunk must divide Wt_core (CB-wrap constraint)"
    gamma_ingest_block = opd._largest_divisor_at_most(wsc, gamma_cap_tiles)
    if regime == "A":
        assert wsc == Wt_core, "Regime A requires WT_SCALE_BLOCK == Wt_core (gamma is never popped)"

    layout = tuple(
        _rsr_cb_layout(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_depth,
            out_depth=out_depth,
            rm_depth=rm_depth,
            gamma_depth=gamma_depth,
            normed_depth=normed_depth,
            fused_reduce=fused_reduce,
            resident_gamma=resident_gamma,
            wr=wr,
            ws=wsc,
            gamma_ingest_block=gamma_ingest_block,
            **common,
        )
    )
    total = sum(p * b for _, p, b, _ in layout)
    assert total <= l1_cb_budget, f"working set {total} > budget {l1_cb_budget}"

    return LabPlan(
        Rt=Rt,
        Wt=Wt,
        Wt_core=Wt_core,
        W_true=W_true,
        W_partial=W_partial,
        num_rows=num_rows,
        is_row_major=is_row_major,
        has_gamma=has_gamma,
        gamma_is_row_major=gamma_is_row_major,
        tile_out=tile_out,
        elem_size=elem_size,
        gamma_elem_size=gamma_elem_size,
        in_tile_bytes=T_in,
        gamma_tile_bytes=T_g,
        interm_dtype=interm_dtype,
        acc_dtype=acc_dtype,
        row_bytes=W_true * elem_size,
        gamma_row_bytes=W_true * gamma_elem_size,
        BLOCK_HT=block_ht,
        WT_REDUCE_BLOCK=wr,
        WT_SCALE_BLOCK=wsc,
        DEST_BLOCK=dest_limit,
        GAMMA_INGEST_BLOCK=gamma_ingest_block,
        IN_BUF_DEPTH=in_depth,
        OUT_BUF_DEPTH=out_depth,
        RM_BUF_DEPTH=rm_depth,
        GAMMA_DEPTH=gamma_depth,
        NORMED_DEPTH=normed_depth,
        regime=regime,
        fused_reduce=fused_reduce,
        resident_gamma=resident_gamma,
        # The DECOMPOSITION arm (Regime C without the fused reduce) runs Regime
        # B's streaming reduce datapath, so it must get B's datapath choice too -
        # otherwise it would silently be compared against B at a DIFFERENT
        # numerical datapath (ReduceTile vs AccumulateViaAdd) and the "one read"
        # attribution would be contaminated by a precision change.
        reduce_via_add=opd._reduce_via_add(
            "B" if (regime == "B" or (regime == "C" and not fused_reduce)) else regime,
            compute_kernel_config,
            interm_dtype,
            W_partial,
            bool(_lever(levers, "reduce_via_add")),
        ),
        num_row_blocks=opd._div_up(Rt, block_ht),
        l1_cb_budget=l1_cb_budget,
        fits_a=fits_a,
        fits_c=fits_c,
        cb_layout=layout,
    )


def plan_summary(p: LabPlan) -> str:
    return (
        f"regime={p.regime} fused={p.fused_reduce} rg={p.resident_gamma} Wt_core={p.Wt_core} BLOCK_HT={p.BLOCK_HT} "
        f"wr={p.WT_REDUCE_BLOCK} ws={p.WT_SCALE_BLOCK} din={p.IN_BUF_DEPTH} dout={p.OUT_BUF_DEPTH} "
        f"dg={p.GAMMA_DEPTH} dn={p.NORMED_DEPTH} dest={p.DEST_BLOCK} rva={p.reduce_via_add} "
        f"blocks={p.num_row_blocks} ws_bytes={p.working_set_bytes()}/{p.l1_cb_budget} "
        f"fitsA={int(p.fits_a)} fitsC={int(p.fits_c)}"
    )


def assert_matches_op_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers=None):
    """Honest-baseline gate: with allow_c=0 the lab must reproduce the op's plan."""
    lev = dict(levers or {})
    lev["allow_c"] = 0
    lab = lab_blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, lev)
    ref = opd.blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers)
    for field in (
        "regime",
        "BLOCK_HT",
        "WT_REDUCE_BLOCK",
        "WT_SCALE_BLOCK",
        "DEST_BLOCK",
        "IN_BUF_DEPTH",
        "OUT_BUF_DEPTH",
        "RM_BUF_DEPTH",
        "GAMMA_INGEST_BLOCK",
        "reduce_via_add",
        "num_row_blocks",
    ):
        assert getattr(lab, field) == getattr(ref, field), f"{field}: lab {getattr(lab,field)} != op {getattr(ref,field)}"
    assert lab.cb_layout == ref.cb_layout, f"CB layout drift:\n lab {lab.cb_layout}\n op  {ref.cb_layout}"
    return ref


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------
def _cb(index, num_pages, page_size, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config, levers=None, out_plan=None
):
    device = input_tensor.device()
    plan = lab_blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers)
    if out_plan is not None:
        out_plan.append(plan)

    grid = device.compute_with_storage_grid_size()
    core_cap = _lever(levers, "active_cores") or opd.ACTIVE_CORE_CAP
    if core_cap:
        rows = max(1, opd._div_up(core_cap, grid.x))
        grid = ttnn.CoreCoord(grid.x, min(grid.y, rows))

    row_wise = bool(_lever(levers, "row_wise"))
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        bpc1,
        bpc2,
    ) = ttnn.split_work_to_cores(grid, plan.num_row_blocks, row_wise)
    cores = ttnn.grid_to_cores(num_cores, grid.x, grid.y, row_wise)

    fmt_of_kind = {
        "in": input_tensor.dtype,
        "out": output_tensor.dtype,
        "gamma": gamma.dtype if plan.has_gamma else input_tensor.dtype,
        "interm": plan.interm_dtype,
        "acc": plan.acc_dtype,
        "bf16": ttnn.bfloat16,
    }
    cbs = [
        _cb(index, num_pages, page_bytes, fmt_of_kind[kind], all_cores)
        for index, num_pages, page_bytes, kind in plan.cb_layout
    ]

    geometry_ct_args = [
        1 if plan.is_row_major else 0,  # 0
        REGIME_CODE[plan.regime],  # 1  (0=B, 1=A, 2=C)
        1 if plan.has_gamma else 0,  # 2
        1 if plan.gamma_is_row_major else 0,  # 3
        plan.Wt_core,  # 4
        plan.W_partial,  # 5
        plan.BLOCK_HT,  # 6
        plan.WT_REDUCE_BLOCK,  # 7
        plan.WT_SCALE_BLOCK,  # 8
        plan.Rt,  # 9
        plan.num_rows,  # 10
        plan.row_bytes,  # 11
        plan.elem_size,  # 12
        plan.gamma_elem_size,  # 13
        plan.gamma_row_bytes,  # 14
        plan.DEST_BLOCK,  # 15
        plan.gamma_tile_bytes,  # 16
        plan.in_tile_bytes,  # 17
        plan.GAMMA_INGEST_BLOCK,  # 18
        _lever(levers, "barrier_per_block"),  # 19
        _lever(levers, "stub_dm"),  # 20
        _lever(levers, "coalesce"),  # 21
        plan.reduce_via_add,  # 22
        plan.fused_reduce,  # 23 (Regime C reduce datapath)
        plan.resident_gamma,  # 24 (Regime C gamma residency)
    ]

    reader_ct_args = list(geometry_ct_args)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct_args = list(geometry_ct_args)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct_args = list(geometry_ct_args)

    inv_w_bits = opd._f32_bits(1.0 / float(plan.W_true))
    eps_bits = opd._f32_bits(epsilon)

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    start = 0
    for core in cores:
        if core_group_1.contains(core):
            blocks_here = bpc1
        elif core_group_2.contains(core):
            blocks_here = bpc2
        else:
            blocks_here = 0
        reader_rt[core.x][core.y] = [in_addr, gamma_addr, start, blocks_here]
        writer_rt[core.x][core.y] = [out_addr, start, blocks_here]
        compute_rt[core.x][core.y] = [inv_w_bits, eps_bits, start, blocks_here]
        start += blocks_here

    zone_def = [("RMSN_NO_ZONES", "1")] if _lever(levers, "no_zones") else []
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        defines=list(zone_def),
        runtime_args=reader_rt,
        config=(
            ttnn.ReaderConfigDescriptor()
            if _lever(levers, "noc_split")
            else ttnn.DataMovementConfigDescriptor(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
        ),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        defines=list(zone_def),
        runtime_args=writer_rt,
        config=(
            ttnn.WriterConfigDescriptor()
            if _lever(levers, "noc_split")
            else ttnn.DataMovementConfigDescriptor(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        ),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        defines=(
            list(zone_def) + ([("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if _lever(levers, "stub_compute") else [])
        ),
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=cbs
    )


def lab_rms_norm(
    input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, levers=None, out_plan=None
):
    """The op's entry point, pointed at the lab plan + lab kernels."""
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=None,
    )
    cfg = compute_kernel_config
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    pd = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=cfg,
        levers=levers,
        out_plan=out_plan,
    )
    tensors = [input_tensor] if gamma is None else [input_tensor, gamma]
    tensors.append(output_tensor)
    return ttnn.generic_op(tensors, pd)
