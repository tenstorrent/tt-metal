# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pipeline_overlap lab: an rms_norm blocking plan with the depth knobs EXPOSED.

WHY THIS FILE EXISTS
--------------------
The op's own `blocking_plan()` decides the W-chunk and the CB depths in a fixed
priority order:

    1. take the COARSEST chunk that divides Wt_core and fits the CB budget,
    2. only THEN try to double-buffer *at that chunk*.

On a wide Regime-B shape step 1 eats the whole budget, so step 2 always fails
and every streaming CB ends up at depth 1 -> the reader can never run ahead of
compute.  The op's `_levers` hook can cap the chunk (`wt_block`) and can force
depth 1 (`double_buffer=0`), but it cannot ask for "chunk C at depth D", cannot
put the depth on a CHOSEN CB, and cannot give the reduce pass and the scale pass
different chunks.  Those are exactly the axes this experiment sweeps, so the
solver is re-expressed here with each of them as an explicit knob.

Everything that is NOT one of those knobs is imported from the real op module
(`_cb_layout`'s sibling `_lab_cb_layout` is the one exception - see below), so a
lab arm at the lab defaults must reproduce the op's own plan byte-for-byte.
`assert_matches_op_plan()` checks that on every run; it is the honest-baseline
gate for this whole experiment.

NEW KNOBS (all 0 == "behave exactly like the op")
    wt_reduce / wt_scale : force WT_REDUCE_BLOCK / WT_SCALE_BLOCK separately
    in_depth             : cb_input_tiles / cb_rm_in depth
    out_depth            : cb_output_tiles depth
    gamma_depth          : cb_gamma_tiles depth   (the op hard-codes 1)
    squared_depth        : cb_squared depth       (the op hard-codes 1)
    normed_depth         : cb_normed depth        (the op hard-codes 1)

CB-WRAP INVARIANT (reader kernel, top-of-file comment) is preserved by
`_check_wrap()`: a contiguous N-page access is only legal when the CB's page
count is a multiple of N and the fifo pointer stays N-aligned.  With split
chunks `cb_input_tiles` is accessed at BOTH granularities, so the smaller chunk
must divide the larger and the larger must divide Wt_core.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd
from ttnn.operations.rms_norm.rms_norm import validate

KERNEL_DIR = Path(__file__).parent / "kernels"

LAB_DEFAULTS = dict(opd.LEVER_DEFAULTS)
LAB_DEFAULTS.update(
    {
        "wt_reduce": 0,
        "wt_scale": 0,
        "in_depth": 0,
        "out_depth": 0,
        "gamma_depth": 0,
        "squared_depth": 0,
        "normed_depth": 0,
        "policy_depths": 0,  # (din,dout,dg,dsq,dn): depth-first chunk search
        "policy_split": 0,  # (din,dsq,dg,dn,dout): per-pass chunk search
        "no_zones": 1,  # lab-only: compile the kernels with the zone RAII no-op'd
    }
)


def _lever(levers, name):
    return LAB_DEFAULTS[name] if levers is None else levers.get(name, LAB_DEFAULTS[name])


# --- CB layout, with the three hard-coded depths promoted to knobs ------------
# Mirrors opd._cb_layout() exactly; the ONLY difference is that CB_GAMMA_TILES /
# CB_SQUARED / CB_NORMED carry a depth factor instead of a literal 1.  At the lab
# defaults (all depths 1) the two functions return identical lists, which
# assert_matches_op_plan() verifies on device.
def _lab_cb_layout(
    *,
    regime,
    block_ht,
    in_depth,
    out_depth,
    rm_depth,
    gamma_depth,
    squared_depth,
    normed_depth,
    wr,
    ws,
    Wt_core,
    has_gamma,
    gamma_is_row_major,
    is_row_major,
    tile_out,
    W_partial,
    gamma_ingest_block,
    T_in,
    T_g,
    T_interm,
    T_acc,
    T_bf16,
):
    wmax = max(wr, ws)
    if regime == "A":
        wr = ws = wmax = Wt_core

    layout = [
        (opd.CB_INPUT_TILES, in_depth * block_ht * wmax, T_in, "in"),
        (opd.CB_SUMSQ, (2 if regime == "B" else 1) * block_ht, T_acc, "acc"),
        (opd.CB_RMS_RECIP, block_ht, T_acc, "acc"),
        (opd.CB_REDUCE_SCALER, 2 if (regime == "B" and W_partial) else 1, T_bf16, "bf16"),
    ]
    if regime == "A":
        layout.append((opd.CB_SUMSQ_ACC, block_ht, T_acc, "acc"))
    else:
        layout.append((opd.CB_SQUARED, squared_depth * block_ht * wr, T_interm, "interm"))
    if has_gamma:
        layout.append((opd.CB_GAMMA_TILES, gamma_depth * ws, T_g, "gamma"))
        if gamma_is_row_major:
            layout.append((opd.CB_GAMMA_RM, gamma_ingest_block, T_g, "gamma"))
        layout.append((opd.CB_NORMED, normed_depth * block_ht * ws, T_interm, "interm"))
    layout.append((opd.CB_OUTPUT_TILES, (out_depth if tile_out else 1) * block_ht * ws, T_in, "out"))
    if is_row_major:
        layout.append((opd.CB_RM_IN, rm_depth * wmax, T_in, "in"))
        layout.append((opd.CB_RM_OUT, rm_depth * ws, T_in, "out"))
    return layout


def _bytes(**kw):
    return sum(p * b for _, p, b, _ in _lab_cb_layout(**kw))


def _check_wrap(Wt_core, wr, ws, pages_in, block_ht, regime):
    """The reader/compute/writer CB-WRAP INVARIANT, re-asserted for split chunks."""
    if regime == "A":
        return
    assert Wt_core % wr == 0, f"WT_REDUCE_BLOCK {wr} must divide Wt_core {Wt_core}"
    assert Wt_core % ws == 0, f"WT_SCALE_BLOCK {ws} must divide Wt_core {Wt_core}"
    lo, hi = min(wr, ws), max(wr, ws)
    # cb_input_tiles is accessed at BOTH granularities.  Its page count is a
    # multiple of `hi`; the fifo pointer after the reduce pass has advanced
    # Wt_core (a multiple of hi) pages, so it stays hi-aligned - and therefore
    # lo-aligned - only if lo divides hi.
    assert hi % lo == 0, f"the finer W-chunk {lo} must divide the coarser one {hi}"
    assert pages_in % (block_ht * hi) == 0


def lab_blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers=None):
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

    def ws_bytes(regime, block_ht, in_d, out_d, rm_d, wr, wsc, g_d=1, sq_d=1, n_d=1):
        return _bytes(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_d,
            out_depth=out_d,
            rm_depth=rm_d,
            gamma_depth=g_d,
            squared_depth=sq_d,
            normed_depth=n_d,
            wr=wr,
            ws=wsc,
            gamma_ingest_block=opd._largest_divisor_at_most(wsc, gamma_cap_tiles),
            **common,
        )

    maskless_w = is_row_major or (W_partial == 0)
    fits = ws_bytes("A", 1, 1, 1, 1, Wt_core, Wt_core) <= l1_cb_budget
    regime = "A" if (maskless_w and fits) else "B"

    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    core_cap = _lever(levers, "active_cores") or opd.ACTIVE_CORE_CAP
    if core_cap:
        grid_cores = min(grid_cores, core_cap)
    max_block_ht = max(1, opd._div_up(Rt, max(1, grid_cores)))
    max_block_ht = min(max_block_ht, dest_limit)

    block_ht = 1
    in_depth = out_depth = rm_depth = 1
    gamma_depth = squared_depth = normed_depth = 1

    if regime == "A":
        wr = wsc = Wt_core
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
        # --- LAB: explicit per-pass chunk override -------------------------
        f_wr, f_ws = _lever(levers, "wt_reduce"), _lever(levers, "wt_scale")
        if f_wr:
            wr = f_wr
        if f_ws:
            wsc = f_ws

    # --- LAB: the graduatable POLICY (depth-first chunk search) -------------
    # This is the candidate stated as a rule instead of a number, so the domain
    # sweep can run it on any shape: "pick the COARSEST W-chunk that divides
    # Wt_core and at which the TARGET depth profile still fits L1", i.e. the
    # op's search with the priority of steps 1 and 2 swapped.  Regime A is
    # single-chunk by construction and is deliberately left untouched.
    pol = _lever(levers, "policy_depths")
    pol_split = _lever(levers, "policy_split")
    if regime == "B" and pol:
        p_din, p_dout, p_dg, p_dsq, p_dn = pol
        pick = None
        for cand in range(Wt_core, 0, -1):
            if Wt_core % cand != 0:
                continue
            if ws_bytes("B", 1, p_din, p_dout, p_din, cand, cand, p_dg, p_dsq, p_dn) <= l1_cb_budget:
                pick = cand
                break
        assert pick, "no chunk fits the requested depth profile"
        wr = wsc = pick
    elif regime == "B" and pol_split:
        # Same idea, but the reduce pass and the scale pass get their own
        # granularity: the reduce chunk stays as coarse as the op's (it only
        # pays cb_input + cb_squared), and the scale chunk is then refined until
        # the scale-side depths fit.
        p_din, p_dsq, p_dg, p_dn, p_dout = pol_split
        wr = None
        for cand in range(Wt_core, 0, -1):
            if Wt_core % cand != 0:
                continue
            if ws_bytes("B", 1, p_din, 1, p_din, cand, 1, 1, p_dsq, 1) <= l1_cb_budget:
                wr = cand
                break
        assert wr, "no reduce chunk fits"
        wsc = None
        for cand in range(wr, 0, -1):
            if wr % cand != 0:
                continue
            if ws_bytes("B", 1, p_din, p_dout, p_din, wr, cand, p_dg, p_dsq, p_dn) <= l1_cb_budget:
                wsc = cand
                break
        assert wsc, "no scale chunk fits"

    # --- LAB: explicit depth overrides ------------------------------------
    # Applied BEFORE the BLOCK_HT growth loop so the plan stays deterministic:
    # BLOCK_HT then grows inside whatever budget the requested depths leave,
    # exactly as it would if these depths were the solver's own choice.
    if regime == "B" and pol:
        levers = dict(levers or {})
        levers.update(
            dict(in_depth=pol[0], out_depth=pol[1], gamma_depth=pol[2], squared_depth=pol[3], normed_depth=pol[4])
        )
    elif regime == "B" and pol_split:
        levers = dict(levers or {})
        levers.update(
            dict(
                in_depth=pol_split[0],
                squared_depth=pol_split[1],
                gamma_depth=pol_split[2],
                normed_depth=pol_split[3],
                out_depth=pol_split[4],
            )
        )
    forced_depths = regime == "B" and any(
        _lever(levers, k) for k in ("in_depth", "out_depth", "gamma_depth", "squared_depth", "normed_depth")
    )
    if forced_depths:
        in_depth = _lever(levers, "in_depth") or in_depth
        out_depth = _lever(levers, "out_depth") or out_depth
        gamma_depth = _lever(levers, "gamma_depth") or gamma_depth
        squared_depth = _lever(levers, "squared_depth") or squared_depth
        normed_depth = _lever(levers, "normed_depth") or normed_depth
        rm_depth = _lever(levers, "in_depth") or rm_depth
    elif _lever(levers, "double_buffer"):
        # --- the op's own allocation priority (unchanged) ------------------
        if ws_bytes(regime, block_ht, 2, 2, 2, wr, wsc) <= l1_cb_budget:
            in_depth = out_depth = rm_depth = 2
        elif ws_bytes(regime, block_ht, 2, 1, 2, wr, wsc) <= l1_cb_budget:
            in_depth = rm_depth = 2
        elif ws_bytes(regime, block_ht, 1, 1, 2, wr, wsc) <= l1_cb_budget:
            rm_depth = 2

    def _fits(bh):
        return (
            ws_bytes(regime, bh, in_depth, out_depth, rm_depth, wr, wsc, gamma_depth, squared_depth, normed_depth)
            <= l1_cb_budget
        )

    forced_block_ht = _lever(levers, "block_ht")
    if forced_block_ht:
        max_block_ht = min(max_block_ht, forced_block_ht)
    while block_ht < max_block_ht and _fits(block_ht + 1):
        block_ht += 1
    if not forced_depths and _lever(levers, "double_buffer"):
        while in_depth < 4 and ws_bytes(regime, block_ht, in_depth + 1, out_depth, rm_depth, wr, wsc) <= l1_cb_budget:
            in_depth += 1

    gamma_ingest_block = opd._largest_divisor_at_most(wsc, gamma_cap_tiles)
    if regime == "A":
        assert wsc == Wt_core

    layout = tuple(
        _lab_cb_layout(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_depth,
            out_depth=out_depth,
            rm_depth=rm_depth,
            gamma_depth=gamma_depth,
            squared_depth=squared_depth,
            normed_depth=normed_depth,
            wr=wr,
            ws=wsc,
            gamma_ingest_block=gamma_ingest_block,
            **common,
        )
    )
    total = sum(p * b for _, p, b, _ in layout)
    assert total <= l1_cb_budget, f"lab plan does not fit L1: {total} > {l1_cb_budget}"
    pages_in = next(p for i, p, _, _ in layout if i == opd.CB_INPUT_TILES)
    _check_wrap(Wt_core, wr, wsc, pages_in, block_ht, regime)

    return opd.BlockingPlan(
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
        interm_tile_bytes=T_interm,
        acc_tile_bytes=T_acc,
        bf16_tile_bytes=T_bf16,
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
        regime=regime,
        reduce_via_add=opd._reduce_via_add(
            regime, compute_kernel_config, interm_dtype, W_partial, bool(_lever(levers, "reduce_via_add"))
        ),
        num_row_blocks=opd._div_up(Rt, block_ht),
        l1_cb_budget=l1_cb_budget,
        cb_layout=layout,
    )


# --- program descriptor -------------------------------------------------------
# Structurally a copy of opd.create_program_descriptor.  Two lab-only deltas:
#   * it consumes lab_blocking_plan (the extra depth / split-chunk knobs), and
#   * it passes -DRMSN_NO_ZONES to all three kernels so a chunk-size sweep is not
#     confounded by the per-chunk profiler marker cost.
def lab_create_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config, levers=None):
    device = input_tensor.device()
    compute_kernel_config = opd._apply_precision_levers(compute_kernel_config, levers)
    plan = lab_blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config, levers)

    grid = device.compute_with_storage_grid_size()
    core_cap = _lever(levers, "active_cores") or opd.ACTIVE_CORE_CAP
    if core_cap:
        rows = max(1, opd._div_up(core_cap, grid.x))
        grid = ttnn.CoreCoord(grid.x, min(grid.y, rows))

    row_wise = bool(_lever(levers, "row_wise"))
    (num_cores, all_cores, cg1, cg2, bpc1, bpc2) = ttnn.split_work_to_cores(grid, plan.num_row_blocks, row_wise)
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
        opd._cb(index, num_pages, page_bytes, fmt_of_kind[kind], all_cores)
        for index, num_pages, page_bytes, kind in plan.cb_layout
    ]

    geometry_ct_args = [
        1 if plan.is_row_major else 0,
        1 if plan.regime == "A" else 0,
        1 if plan.has_gamma else 0,
        1 if plan.gamma_is_row_major else 0,
        plan.Wt_core,
        plan.W_partial,
        plan.BLOCK_HT,
        plan.WT_REDUCE_BLOCK,
        plan.WT_SCALE_BLOCK,
        plan.Rt,
        plan.num_rows,
        plan.row_bytes,
        plan.elem_size,
        plan.gamma_elem_size,
        plan.gamma_row_bytes,
        plan.DEST_BLOCK,
        plan.gamma_tile_bytes,
        plan.in_tile_bytes,
        plan.GAMMA_INGEST_BLOCK,
        _lever(levers, "barrier_per_block"),
        _lever(levers, "stub_dm"),
        _lever(levers, "coalesce"),
        plan.reduce_via_add,
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
        if cg1.contains(core):
            blocks_here = bpc1
        elif cg2.contains(core):
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
        defines=(list(zone_def) + ([("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if _lever(levers, "stub_compute") else [])),
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    return (
        ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=cbs),
        plan,
    )


def lab_rms_norm(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, levers=None, out_plan=None):
    """The op's public entry point, re-expressed against the lab descriptor."""
    validate(input_tensor, gamma=gamma, epsilon=epsilon, compute_kernel_config=compute_kernel_config)
    from ttnn.operations.rms_norm.rms_norm import default_compute_kernel_config

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    pd, plan = lab_create_program_descriptor(
        input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=cfg, levers=levers
    )
    if out_plan is not None:
        out_plan.append(plan)
    tensors = [input_tensor] if gamma is None else [input_tensor, gamma]
    tensors.append(output_tensor)
    return ttnn.generic_op(tensors, pd)


def assert_matches_op_plan(input_tensor, gamma, output_tensor, device, cfg, levers=None):
    """Honest-baseline gate: at the lab defaults the lab plan == the op's plan."""
    lab = lab_blocking_plan(input_tensor, gamma, output_tensor, device, cfg, levers)
    ref = opd.blocking_plan(input_tensor, gamma, output_tensor, device, cfg, levers)
    for f in (
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
        "cb_layout",
    ):
        assert getattr(lab, f) == getattr(ref, f), f"lab plan drifted from op plan on {f}: {getattr(lab,f)} vs {getattr(ref,f)}"
    return lab


def plan_summary(plan):
    return (
        f"regime={plan.regime} Wt_core={plan.Wt_core} BLOCK_HT={plan.BLOCK_HT} "
        f"wr={plan.WT_REDUCE_BLOCK} ws={plan.WT_SCALE_BLOCK} DEST={plan.DEST_BLOCK} "
        f"depth(in/out/rm)={plan.IN_BUF_DEPTH}/{plan.OUT_BUF_DEPTH}/{plan.RM_BUF_DEPTH} "
        f"ws_bytes={plan.working_set_bytes()}/{plan.l1_cb_budget} "
        f"cbs={[(i,p) for i,p,_,_ in plan.cb_layout]}"
    )
