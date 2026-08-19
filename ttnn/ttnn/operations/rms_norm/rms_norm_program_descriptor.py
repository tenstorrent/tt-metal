# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for rms_norm (op_design.md "Blocking Model" / "Work Distribution").

Everything blocking-related is decided in exactly ONE place — `blocking_plan()` —
and every downstream quantity (CB page counts, kernel CT/RT args, loop trip
counts, grid sizing) reads a field off the returned frozen dataclass.  Turning a
knob is a one-line change inside `blocking_plan`.

Axes:
    Rt  (tile-rows)     INDEPENDENT  -> split across the whole core grid,
                                        block factor BLOCK_HT.
    Wt  (tile-columns)  DEPENDENT    -> one core owns a full row in Phase 0;
                                        block factors WT_REDUCE_BLOCK /
                                        WT_SCALE_BLOCK bound L1 in Regime B.
    gamma[W]            REUSE-SHARED -> replicated per core (Lamp L2 = mcast).

Two compute regimes, selected by a pinned host predicate:
    A  RESIDENT-FUSED    single DRAM read, fused sum-of-squares, no mask.
    B  STREAMING-MASKED  two DRAM reads, chunked square+accumulating reduce with
                         a partial scaler that zeroes the pad columns.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# Single source of truth for the non-CB L1 reservation (op_design.md).
L1_RESERVED_BYTES = 256 * 1024

# Core-count knob.  None == the full compute grid (Phase 0 default, lever A0).
ACTIVE_CORE_CAP: Optional[int] = None

# --- Circular buffer slots (semantic names; the number is just the slot) ------
CB_INPUT_TILES = 0
CB_GAMMA_TILES = 1
CB_REDUCE_SCALER = 2
CB_SQUARED = 3
CB_SUMSQ = 4
CB_RMS_RECIP = 5
CB_NORMED = 6
CB_OUTPUT_TILES = 7
CB_RM_IN = 8
CB_RM_OUT = 9
CB_SUMSQ_ACC = 10


def _div_up(a: int, b: int) -> int:
    """Ceiling division (ttnn exposes no div_up binding in this tree)."""
    return (a + b - 1) // b


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def _dest_limit(compute_kernel_config) -> int:
    """Host-side mirror of kernel_lib's DEST_AUTO_LIMIT (dest_helpers.hpp).

    The kernel itself always clamps against the real constant; this is only used
    to cap the host-chosen knobs so the plan is self-consistent.
    """
    full_sync = bool(getattr(compute_kernel_config, "dst_full_sync_en", False))
    base = 16 if full_sync else 8
    return base // 2 if bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True)) else base


# ---------------------------------------------------------------------------
# Alignment-aware tile geometry — ceil, PER IMAGE (never floor / //).
# ---------------------------------------------------------------------------


def tile_geometry(shape, is_row_major: bool):
    """Return (Rt, Wt, W_true, W_partial, num_rows).

    TILE layout: each (..., H, W) image is tile-padded independently, so
    Rt = batch * ceil(H / 32).  Writing (batch*H)//32 is wrong for every
    multi-batch h_non_aligned shape.

    ROW_MAJOR layout: the buffer is a flat list of `batch * H` sticks and the
    per-row reduction has no cross-row term, so tile-rows are just groups of 32
    consecutive sticks -> Rt = ceil(batch * H / 32).
    """
    W_true = shape[-1]
    H = shape[-2]
    batch = 1
    for d in shape[:-2]:
        batch *= d
    Wt = _div_up(W_true, TILE_DIM)
    num_rows = batch * H
    if is_row_major:
        Rt = _div_up(num_rows, TILE_DIM)
    else:
        Rt = batch * _div_up(H, TILE_DIM)
    return Rt, Wt, W_true, W_true % TILE_DIM, num_rows


@dataclass(frozen=True)
class BlockingPlan:
    # --- geometry -----------------------------------------------------------
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
    # --- byte geometry ------------------------------------------------------
    elem_size: int
    gamma_elem_size: int
    in_tile_bytes: int
    gamma_tile_bytes: int
    f32_tile_bytes: int
    bf16_tile_bytes: int
    row_bytes: int
    gamma_row_bytes: int
    # --- knobs (every one of these is a tunable, never an inlined literal) ---
    BLOCK_HT: int
    WT_REDUCE_BLOCK: int
    WT_REDUCE_TAIL: int
    WT_SCALE_BLOCK: int
    WT_SCALE_TAIL: int
    DEST_BLOCK: int
    IN_BUF_DEPTH: int
    OUT_BUF_DEPTH: int
    RM_BUF_DEPTH: int
    # --- derived ------------------------------------------------------------
    regime: str
    num_row_blocks: int
    l1_cb_budget: int


def _working_set_bytes(
    *,
    regime: str,
    block_ht: int,
    in_depth: int,
    out_depth: int,
    rm_depth: int,
    wr: int,
    ws: int,
    Wt_core: int,
    has_gamma: bool,
    gamma_is_row_major: bool,
    is_row_major: bool,
    tile_out: bool,
    W_partial: int,
    T_in: int,
    T_g: int,
    T_f32: int,
    T_bf16: int,
) -> int:
    """Total L1 bytes the CB set costs for this knob assignment.

    All CBs are statically allocated for the whole program, so this is a SUM
    over every CB the configuration creates (not a per-phase max).
    """
    wmax = max(wr, ws)
    if regime == "A":
        wr = ws = wmax = Wt_core

    total = 0
    # cb_input_tiles
    total += in_depth * block_ht * wmax * T_in
    # cb_sumsq (also the cross-chunk accumulator in Regime B -> 2x headroom)
    total += 2 * block_ht * T_f32
    # cb_rms_recip
    total += block_ht * T_f32
    # cb_reduce_scaler (bfloat16, mandatory format) — both regimes need the
    # within-tile REDUCE_ROW finalize.
    total += (2 if (regime == "B" and W_partial) else 1) * T_bf16
    if regime == "A":
        # cb_sumsq_acc — sum_of_squares' element-wise tile accumulator, which the
        # finalize reduce collapses along W.
        total += block_ht * T_f32
    else:
        # cb_squared — sequential-helper intermediate: full block per call
        total += block_ht * wr * T_in
    if has_gamma:
        total += ws * T_g  # cb_gamma_tiles
        total += block_ht * ws * T_in  # cb_normed
    # cb_output_tiles — streamed to the writer on the TILE path, but feeds the
    # sequential untilize helper on the RM path (must hold the full block).
    total += (out_depth if tile_out else 1) * block_ht * ws * T_in
    if is_row_major:
        total += rm_depth * wmax * T_in  # cb_rm_in  (one tile-row of sticks)
        total += rm_depth * ws * T_in  # cb_rm_out (one tile-row of tiles)
    return total


def blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config) -> BlockingPlan:
    """The ONLY place block factors, buffer depths and the regime are decided."""
    shape = list(input_tensor.shape)
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    Rt, Wt, W_true, W_partial, num_rows = tile_geometry(shape, is_row_major)

    Wt_core = Wt  # Phase 0: no W split across cores.  Lamp L1 sets Wt/num_w_cores.

    has_gamma = gamma is not None
    gamma_is_row_major = bool(has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT)

    elem_size = input_tensor.element_size()
    gamma_elem_size = gamma.element_size() if has_gamma else elem_size
    T_in = ttnn.tile_size(input_tensor.dtype)
    T_g = ttnn.tile_size(gamma.dtype) if has_gamma else T_in
    T_f32 = ttnn.tile_size(ttnn.float32)
    T_bf16 = ttnn.tile_size(ttnn.bfloat16)

    tile_out = not is_row_major
    l1_cb_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVED_BYTES

    dest_limit = _dest_limit(compute_kernel_config)

    common = dict(
        Wt_core=Wt_core,
        has_gamma=has_gamma,
        gamma_is_row_major=gamma_is_row_major,
        is_row_major=is_row_major,
        tile_out=tile_out,
        W_partial=W_partial,
        T_in=T_in,
        T_g=T_g,
        T_f32=T_f32,
        T_bf16=T_bf16,
    )

    def ws_bytes(regime, block_ht, in_depth, out_depth, rm_depth, wr, wsc):
        return _working_set_bytes(
            regime=regime,
            block_ht=block_ht,
            in_depth=in_depth,
            out_depth=out_depth,
            rm_depth=rm_depth,
            wr=wr,
            ws=wsc,
            **common,
        )

    # --- Regime selection (pinned predicate, op_design.md) ------------------
    #  (1) can the reduce see the padded columns without a mask?
    #      RM   -> the reader zero-fills every stick's pad tail: pad is exactly 0.
    #      TILE -> the pad lives in DRAM and may be poisoned: mask mandatory.
    #  (2) does the MINIMAL resident working set fit the CB budget?
    maskless_w = is_row_major or (W_partial == 0)
    fits = ws_bytes("A", 1, 1, 1, 1, Wt_core, Wt_core) <= l1_cb_budget
    regime = "A" if (maskless_w and fits) else "B"

    # --- Grid / core count (needed to cap BLOCK_HT so cores are not starved) -
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    if ACTIVE_CORE_CAP is not None:
        grid_cores = min(grid_cores, ACTIVE_CORE_CAP)
    # Coarsest useful row-block: any coarser and some cores get no work at all.
    max_block_ht = max(1, _div_up(Rt, max(1, grid_cores)))
    max_block_ht = min(max_block_ht, dest_limit)

    block_ht = 1
    in_depth = out_depth = rm_depth = 1

    if regime == "A":
        wr = wsc = Wt_core
    else:
        # Coarsest chunk of the dependent axis that still fits L1.  Never 1 by
        # default — 1 is only ever the *output* of this search.
        wr = wsc = 1
        for cand in range(Wt_core, 0, -1):
            if ws_bytes("B", 1, 1, 1, 1, cand, cand) <= l1_cb_budget:
                wr = wsc = cand
                break

    # Allocation priority (movement-dominated op: overlap beats amortization):
    #   1. double-buffer the streaming CBs (lever C16, measured 2.78x)
    #   2. grow BLOCK_HT (per-block-overhead amortization)
    #   3. grow IN_BUF_DEPTH further
    if ws_bytes(regime, block_ht, 2, 2, 2, wr, wsc) <= l1_cb_budget:
        in_depth = out_depth = rm_depth = 2
    elif ws_bytes(regime, block_ht, 2, 1, 2, wr, wsc) <= l1_cb_budget:
        in_depth = rm_depth = 2
    elif ws_bytes(regime, block_ht, 1, 1, 2, wr, wsc) <= l1_cb_budget:
        rm_depth = 2

    while (
        block_ht < max_block_ht
        and ws_bytes(regime, block_ht + 1, in_depth, out_depth, rm_depth, wr, wsc) <= l1_cb_budget
    ):
        block_ht += 1

    while in_depth < 4 and ws_bytes(regime, block_ht, in_depth + 1, out_depth, rm_depth, wr, wsc) <= l1_cb_budget:
        in_depth += 1

    # Chunk tails (the last chunk of the dependent axis may be narrower).
    num_reduce_chunks = _div_up(Wt_core, wr)
    wr_tail = Wt_core - (num_reduce_chunks - 1) * wr
    num_scale_chunks = _div_up(Wt_core, wsc)
    ws_tail = Wt_core - (num_scale_chunks - 1) * wsc

    # R5: cb_gamma_tiles is never popped in Regime A, so one pass-B call must
    # span every gamma column from the CB front.
    if regime == "A":
        assert wsc == Wt_core, "Regime A requires WT_SCALE_BLOCK == Wt_core (gamma is never popped)"

    return BlockingPlan(
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
        f32_tile_bytes=T_f32,
        bf16_tile_bytes=T_bf16,
        row_bytes=W_true * elem_size,
        gamma_row_bytes=W_true * gamma_elem_size,
        BLOCK_HT=block_ht,
        WT_REDUCE_BLOCK=wr,
        WT_REDUCE_TAIL=wr_tail,
        WT_SCALE_BLOCK=wsc,
        WT_SCALE_TAIL=ws_tail,
        DEST_BLOCK=dest_limit,
        IN_BUF_DEPTH=in_depth,
        OUT_BUF_DEPTH=out_depth,
        RM_BUF_DEPTH=rm_depth,
        regime=regime,
        num_row_blocks=_div_up(Rt, block_ht),
        l1_cb_budget=l1_cb_budget,
    )


def _cb(index, num_pages, page_size, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor,
    gamma,
    output_tensor,
    *,
    epsilon: float,
    compute_kernel_config,
) -> "ttnn.ProgramDescriptor":
    device = input_tensor.device()
    plan = blocking_plan(input_tensor, gamma, output_tensor, device, compute_kernel_config)

    # ---------------- work distribution over the INDEPENDENT axis ------------
    grid = device.compute_with_storage_grid_size()
    if ACTIVE_CORE_CAP is not None:
        # Truncate the grid row-wise so the cap keeps the DRAM-facing spread.
        rows = max(1, _div_up(ACTIVE_CORE_CAP, grid.x))
        grid = ttnn.CoreCoord(grid.x, min(grid.y, rows))

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, plan.num_row_blocks, True)

    cores = ttnn.grid_to_cores(num_cores, grid.x, grid.y, True)

    # ---------------- circular buffers ---------------------------------------
    wmax = max(plan.WT_REDUCE_BLOCK, plan.WT_SCALE_BLOCK)
    cbs = [
        _cb(
            CB_INPUT_TILES,
            plan.IN_BUF_DEPTH * plan.BLOCK_HT * wmax,
            plan.in_tile_bytes,
            input_tensor.dtype,
            all_cores,
        ),
        _cb(CB_SUMSQ, 2 * plan.BLOCK_HT, plan.f32_tile_bytes, ttnn.float32, all_cores),
        _cb(CB_RMS_RECIP, plan.BLOCK_HT, plan.f32_tile_bytes, ttnn.float32, all_cores),
        _cb(
            CB_OUTPUT_TILES,
            (plan.OUT_BUF_DEPTH if plan.tile_out else 1) * plan.BLOCK_HT * plan.WT_SCALE_BLOCK,
            plan.in_tile_bytes,
            output_tensor.dtype,
            all_cores,
        ),
    ]
    cbs.append(
        _cb(
            CB_REDUCE_SCALER,
            2 if (plan.regime == "B" and plan.W_partial) else 1,
            plan.bf16_tile_bytes,
            ttnn.bfloat16,
            all_cores,
        )
    )
    if plan.regime == "A":
        cbs.append(_cb(CB_SUMSQ_ACC, plan.BLOCK_HT, plan.f32_tile_bytes, ttnn.float32, all_cores))
    else:
        cbs.append(
            _cb(
                CB_SQUARED,
                plan.BLOCK_HT * plan.WT_REDUCE_BLOCK,
                plan.in_tile_bytes,
                input_tensor.dtype,
                all_cores,
            )
        )
    if plan.has_gamma:
        cbs.append(_cb(CB_GAMMA_TILES, plan.WT_SCALE_BLOCK, plan.gamma_tile_bytes, gamma.dtype, all_cores))
        cbs.append(
            _cb(
                CB_NORMED,
                plan.BLOCK_HT * plan.WT_SCALE_BLOCK,
                plan.in_tile_bytes,
                input_tensor.dtype,
                all_cores,
            )
        )
    if plan.is_row_major:
        cbs.append(_cb(CB_RM_IN, plan.RM_BUF_DEPTH * wmax, plan.in_tile_bytes, input_tensor.dtype, all_cores))
        cbs.append(
            _cb(
                CB_RM_OUT,
                plan.RM_BUF_DEPTH * plan.WT_SCALE_BLOCK,
                plan.in_tile_bytes,
                output_tensor.dtype,
                all_cores,
            )
        )

    # ---------------- compile-time args --------------------------------------
    # One shared geometry prefix so reader / writer / compute cannot drift.
    geometry_ct_args = [
        1 if plan.is_row_major else 0,  # 0  IS_ROW_MAJOR
        1 if plan.regime == "A" else 0,  # 1  REGIME_A
        1 if plan.has_gamma else 0,  # 2  HAS_GAMMA
        1 if plan.gamma_is_row_major else 0,  # 3  GAMMA_IS_ROW_MAJOR
        plan.Wt_core,  # 4
        plan.W_partial,  # 5
        plan.BLOCK_HT,  # 6
        plan.WT_REDUCE_BLOCK,  # 7
        plan.WT_REDUCE_TAIL,  # 8
        plan.WT_SCALE_BLOCK,  # 9
        plan.WT_SCALE_TAIL,  # 10
        plan.Rt,  # 11
        plan.num_rows,  # 12
        plan.row_bytes,  # 13
        plan.elem_size,  # 14
        plan.gamma_elem_size,  # 15
        plan.gamma_row_bytes,  # 16
        plan.DEST_BLOCK,  # 17
        plan.gamma_tile_bytes,  # 18
        plan.in_tile_bytes,  # 19
    ]

    # ---------------- reader --------------------------------------------------
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

    inv_w_bits = _f32_bits(1.0 / float(plan.W_true))
    eps_bits = _f32_bits(epsilon)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    start = 0
    for core in cores:
        if core_group_1.contains(core):
            blocks_here = blocks_per_core_g1
        elif core_group_2.contains(core):
            blocks_here = blocks_per_core_g2
        else:
            blocks_here = 0
        reader_rt[core.x][core.y] = [in_addr, gamma_addr, start, blocks_here]
        writer_rt[core.x][core.y] = [out_addr, start, blocks_here]
        compute_rt[core.x][core.y] = [inv_w_bits, eps_bits, start, blocks_here]
        start += blocks_here

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        # Pass-through: the caller's descriptor is handed over verbatim.
        config=compute_kernel_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
