# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR EVERY BLOCK / DEPTH / GRID KNOB
(op_design.md section 1.4).  Kernels receive the knobs as compile-time /
runtime args and never re-derive them; no block or chunk count is restated as a
second literal anywhere.

Knob map (all tunable parameters, none inlined):

  primary (the only hand-set numbers)
    L1_SAFETY_FRACTION      fraction of usable per-core L1 the CBs may take
    CB_RM_STAGE_DEPTH       depth of the ROW_MAJOR stick staging CBs
    GRID_W                  cores along `width` (Lamp L1 lives at its trivial 1)
    CB_X_DEPTH / CB_OUT_DEPTH   cross-processor CB depth (2 when the producer /
                            consumer is a dataflow kernel, 1 when it is a
                            sequential compute helper — depth buys nothing then)
    REDUCE_BULK             reduce input policy (BulkWaitBulkPop vs per-tile)

  derived block factors
    BLOCK_ROWS   tile-rows per compute block = min(per-core assignment,
                 the coarsest chunk that fits the L1 budget)
    WT_CHUNK     width tiles per compute block = Wt in the RESIDENT regime;
                 the coarsest DIVISOR of Wt that fits L1 in STREAM
    NUM_W_CHUNKS = Wt // WT_CHUNK      (X_RESIDENT == GAMMA_RESIDENT == (== 1))

Deviations from op_design.md section 1.4 (advisory: CB sizing / knob selection;
the scheme, topology, work split and helper mapping are unchanged):

  D1  WT_CHUNK is constrained to a DIVISOR of Wt, so every width chunk is the
      same size.  Three mechanisms in the chosen helper set require a uniform
      chunk and would otherwise need a ragged-tail special case:
        * compute_kernel_lib::tilize / untilize take `block_width_tiles` as a
          COMPILE-TIME template parameter (tilize_helpers.hpp:188);
        * reduce()'s BulkWaitBulkPop asserts
          `num_pages(cb_in) % cols == 0` (reduce_helpers_compute.inl:698-699);
        * a multi-page cb_reserve_back / get_write_ptr batch must not straddle
          the CB ring, i.e. the ring size must be a multiple of the push unit.
      WT_CHUNK is still the coarsest value the L1 budget allows (largest
      admissible divisor), so the knob is not collapsed.
  D2  The STREAM chunk-size solve counts the ROW_MAJOR staging CBs at
      WT_CHUNK tiles (what is actually allocated), not at Wt.
  D3  accumulate_reduce_block() does not expose reduce()'s ReduceFp32Mode
      template slot (streaming_reduce_helpers.hpp:47-61), so the reduce runs at
      the default Fast mode.  fp32 DEST accumulation still comes from
      fp32_dest_acc_en=True.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# ---------------------------------------------------------------------------
# Primary knobs — the only hand-set numbers in this op.
# ---------------------------------------------------------------------------

# Fraction of usable per-core L1 the CBs may occupy.  Lower it if CB-OOM shows
# up.  Everything about block size derives from the resulting byte budget.
L1_SAFETY_FRACTION = 0.85

# Depth of the ROW_MAJOR stick staging CBs (reader <-> tilize overlap).
CB_RM_STAGE_DEPTH = 2

# Cores along the `width` axis.  Trivial value 1 in Phase 0 = one core owns the
# whole width of every row it owns; Lamp L1 turns this knob up and adds the
# cross-core partial-sum combine.
GRID_W = 1

# reduce() input policy knob: 1 = BulkWaitBulkPop (bulk wait/indexed/bulk pop),
# 0 = WaitAndPopPerTile.  Bulk is the coarse default (op_design.md section 1.4).
REDUCE_BULK = 1


# ---------------------------------------------------------------------------
# Small host helpers (ttnn exposes no div_up / round_up binding).
# ---------------------------------------------------------------------------


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Coarsest d with n % d == 0 and d <= cap (D1)."""
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _f32_bits(v: float) -> int:
    return struct.unpack("I", struct.pack("f", float(v)))[0]


# ---------------------------------------------------------------------------
# Circular-buffer slots (semantic names; the number is just the slot).
# ---------------------------------------------------------------------------

CB_INPUT_STICKS = 0  # ROW_MAJOR only: padded row-major staging of x
CB_INPUT_TILES = 1  # x tiles (reader for TILE, tilize for ROW_MAJOR)
CB_X_SQUARED = 2  # x^2 tiles, pass A
CB_SCALER = 3  # reduce scaler (bf16, value 1.0) + partial scaler
CB_ROW_STAT = 4  # fp32: sum(x^2) accumulator -> in-place 1/rms
CB_GAMMA_STICKS = 5  # ROW_MAJOR gamma only
CB_GAMMA_TILES = 6  # gamma tiles (row 0 valid)
CB_NORMALIZED = 7  # x * (1/rms), only when gamma is present
CB_OUTPUT_TILES = 8  # output tiles
CB_OUTPUT_STICKS = 9  # ROW_MAJOR only: untilized row-major staging of out


def _core_range_set_full_grid(device):
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def _cores_in(core_range_set):
    try:
        return list(ttnn.corerange_to_cores(core_range_set, None, True))
    except Exception:
        return []


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    *,
    gamma: "ttnn.Tensor" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
) -> "ttnn.ProgramDescriptor":
    device = input_tensor.device()
    shape = list(input_tensor.shape)

    is_tile = input_tensor.layout == ttnn.TILE_LAYOUT
    has_gamma = gamma is not None
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    # ---- geometry (alignment-aware: ceil everywhere, per image) ------------
    W = shape[-1]
    Wt = _div_up(W, TILE_DIM)
    partial_w = W % TILE_DIM  # 0 => tile-aligned width

    if is_tile:
        # Every (..., H, W) image is tile-padded independently.
        Rt = _prod(shape[:-2]) * _div_up(shape[-2], TILE_DIM)
        R_rm = 0  # unused
    else:
        # ROW_MAJOR has no implicit H padding: all images' rows are contiguous.
        R_rm = _prod(shape[:-1])
        Rt = _div_up(R_rm, TILE_DIM)

    elem_bytes = input_tensor.element_size()
    gamma_elem_bytes = gamma.element_size() if has_gamma else 0

    # ---- buffer-depth knobs ------------------------------------------------
    # Depth > 1 only buys overlap when the producer/consumer pair spans two
    # processors.  On the ROW_MAJOR path cb_input_tiles is produced by the
    # `tilize` compute helper and cb_output_tiles is consumed by `untilize`;
    # sequential compute helpers own all three TRISCs and cannot pipeline, so
    # those CBs drop to depth 1 (and must instead hold a whole block — R5).
    cb_x_depth = 2 if is_tile else 1
    cb_out_depth = 2 if is_tile else 1

    # ---- L1 byte budget (derived from the device, never a literal) --------
    l1_budget = int(ttnn.get_max_worker_l1_unreserved_size() * L1_SAFETY_FRACTION)

    bt = ttnn.tile_size(input_tensor.dtype)  # x / x^2 / normalized / out
    gt = ttnn.tile_size(gamma.dtype) if has_gamma else 0
    st = ttnn.tile_size(ttnn.bfloat16)  # scaler CB (R4: value exactly 1.0)
    ft = ttnn.tile_size(ttnn.float32)  # cb_row_stat

    # tiles-per-block multiplier over the block-scoped, input-dtype CBs:
    #   cb_input_tiles + cb_x_squared + cb_normalized + cb_output_tiles
    cb_block_mult = cb_x_depth + 1 + (1 if has_gamma else 0) + cb_out_depth

    scaler_bytes = st * (2 if partial_w else 1)

    # ---- regime selection (op_design.md section 4.2; pure function of
    # (layout, dtypes, gamma format, Wt, budget) — device-independent) -------
    fixed_resident = (
        (Wt * gt)  # cb_gamma_tiles
        + (Wt * gt if gamma_is_rm else 0)  # cb_gamma_sticks
        + (2 * CB_RM_STAGE_DEPTH * Wt * bt if not is_tile else 0)  # stick staging
        + scaler_bytes
    )
    per_tilerow_bytes = Wt * bt * cb_block_mult + ft  # + one cb_row_stat page
    block_rows_l1_max = max(0, (l1_budget - fixed_resident) // per_tilerow_bytes)

    # ---- cross-core split of the independent `row` axis: SIZE and COUNT ----
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(
        _core_range_set_full_grid(device), Rt, True
    )  # row_wise=True is mandatory
    max_rows_per_core = max(rows_per_core_g1, rows_per_core_g2)

    if block_rows_l1_max >= 1:
        # RESIDENT: the whole row is resident; take the coarsest row block that
        # fits, i.e. the entire per-core assignment whenever it fits.
        block_rows = min(max_rows_per_core, block_rows_l1_max)
        wt_chunk = Wt
        num_w_chunks = 1
    else:
        # STREAM: one tile-row's width does not fit -> chunk `width` (an L1
        # fallback, not a parallelization).  x is re-read in pass B.
        block_rows = 1
        per_chunk_tile_bytes = (
            bt * cb_block_mult
            + (gt * (2 if gamma_is_rm else 1) if has_gamma else 0)
            + (2 * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)  # D2
        )
        fixed_stream = scaler_bytes + ft
        wt_chunk_l1_max = max(1, (l1_budget - fixed_stream) // per_chunk_tile_bytes)
        wt_chunk = _largest_divisor_at_most(Wt, wt_chunk_l1_max)  # D1
        num_w_chunks = Wt // wt_chunk

    x_resident = num_w_chunks == 1  # == GAMMA_RESIDENT (op_design.md section 1.4)

    # ---- per-core assignment: (row_start, row_count) prefix sum -----------
    cores_g1 = _cores_in(core_group_1)
    cores_g2 = _cores_in(core_group_2)
    assignment = []  # [(core, row_start, row_count)]
    row_cursor = 0
    for core in cores_g1:
        assignment.append((core, row_cursor, rows_per_core_g1))
        row_cursor += rows_per_core_g1
    for core in cores_g2:
        assignment.append((core, row_cursor, rows_per_core_g2))
        row_cursor += rows_per_core_g2
    assert row_cursor == Rt, f"rms_norm: work split covers {row_cursor} of {Rt} tile-rows"
    assert len(assignment) == num_cores, f"rms_norm: {len(assignment)} cores assigned, expected {num_cores}"

    # ---- circular buffers -------------------------------------------------
    # Every page count below is a function of the block/depth knobs only.
    # cb_gamma_* is the one place Wt appears; it *is* the gamma tensor's extent
    # and is bounded by the same L1 predicate through `fixed_resident`
    # (collapsing to WT_CHUNK in the STREAM regime).
    cbs = []
    if not is_tile:
        cbs.append(_cb(CB_INPUT_STICKS, bt, CB_RM_STAGE_DEPTH * wt_chunk, input_tensor.dtype, all_cores))
        cbs.append(_cb(CB_OUTPUT_STICKS, bt, CB_RM_STAGE_DEPTH * wt_chunk, output_tensor.dtype, all_cores))
    cbs.append(_cb(CB_INPUT_TILES, bt, cb_x_depth * block_rows * wt_chunk, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_X_SQUARED, bt, block_rows * wt_chunk, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_SCALER, st, 2 if partial_w else 1, ttnn.bfloat16, all_cores))
    cbs.append(_cb(CB_ROW_STAT, ft, block_rows, ttnn.float32, all_cores))
    if has_gamma:
        if gamma_is_rm:
            cbs.append(_cb(CB_GAMMA_STICKS, gt, wt_chunk, gamma.dtype, all_cores))
        cbs.append(_cb(CB_GAMMA_TILES, gt, wt_chunk, gamma.dtype, all_cores))
        cbs.append(_cb(CB_NORMALIZED, bt, block_rows * wt_chunk, input_tensor.dtype, all_cores))
    cbs.append(_cb(CB_OUTPUT_TILES, bt, cb_out_depth * block_rows * wt_chunk, output_tensor.dtype, all_cores))

    # ---- reader -----------------------------------------------------------
    reader_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        block_rows,  # 4  BLOCK_ROWS
        partial_w,  # 5  PARTIAL_W (0 => aligned)
        1 if has_gamma else 0,  # 6  HAS_GAMMA
        1 if gamma_is_rm else 0,  # 7  GAMMA_IS_RM
        elem_bytes,  # 8  input element bytes
        gamma_elem_bytes,  # 9  gamma element bytes
        R_rm,  # 10 total ROW_MAJOR sticks (0 for TILE)
        W,  # 11 logical width (elements)
    ]
    n_reader_scalars = len(reader_ct_args)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    assert n_reader_scalars == 12

    # ---- writer -----------------------------------------------------------
    writer_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        Wt,  # 1  WT
        wt_chunk,  # 2  WT_CHUNK
        num_w_chunks,  # 3  NUM_W_CHUNKS
        block_rows,  # 4  BLOCK_ROWS
        elem_bytes,  # 5  output element bytes
        R_rm,  # 6  total ROW_MAJOR sticks (0 for TILE)
        W,  # 7  logical width (elements)
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- compute ----------------------------------------------------------
    compute_ct_args = [
        1 if is_tile else 0,  # 0  IS_TILE
        wt_chunk,  # 1  WT_CHUNK
        num_w_chunks,  # 2  NUM_W_CHUNKS
        block_rows,  # 3  BLOCK_ROWS
        partial_w,  # 4  PARTIAL_W
        1 if has_gamma else 0,  # 5  HAS_GAMMA
        1 if gamma_is_rm else 0,  # 6  GAMMA_IS_RM
        _f32_bits(1.0 / float(W)),  # 7  INV_W (raw fp32 bits) -- R1/R4: logical W
        _f32_bits(epsilon),  # 8  EPS (raw fp32 bits)
        REDUCE_BULK,  # 9  reduce input policy knob
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    for core, row_start, row_count in assignment:
        reader_rt[core.x][core.y] = [x_addr, g_addr, row_start, row_count]
        writer_rt[core.x][core.y] = [out_addr, row_start, row_count]
        compute_rt[core.x][core.y] = [row_count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,  # passed through unmodified
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
