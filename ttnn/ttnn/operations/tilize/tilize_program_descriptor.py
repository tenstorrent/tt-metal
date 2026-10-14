# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor for the `interleaved_row_split` regime (op_design.md).

Work = the R x C grid of OUTPUT tiles. Tile-rows are split across the whole
compute grid (`split_work_to_cores(..., row_wise=True)`); every core owns the
rectangle (row_start_tile, core_rows_tiles, col_start_tile, core_cols_tiles).
Inside a core the rectangle is walked as blocks of
block_height_tiles x block_width_tiles; every block knob is defined once here
and all CB page counts / trip counts derive from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names) ---------------------------------------------
CB_RM_STICKS = 0  # reader -> compute: RM stick segments, tile-sized pages
CB_RM_STICKS_TAIL = 1  # same, for the ragged last column block (own ring quantum)
CB_STICK_STAGING = 2  # reader-private scratch: bank-grouped sticks of one tile-row
CB_PAD_FILL = 3  # reader-private: one stick segment of the pad value (source of local fill copies)
CB_TILES_OUT = 16  # compute -> writer: tiled output pages
CB_TILES_OUT_TAIL = 17  # same, for the ragged last column block

# --- Block knobs (single source of truth) -----------------------------------
IN_DEPTH_TILE_ROWS = 2  # cb_rm_sticks window, in tile-rows (double buffer)
OUT_DEPTH_TILE_ROWS = 2  # cb_tiles_out window, in tile-rows (double buffer)
L1_BLOCK_BUDGET_BYTES = 512 * 1024  # per-core budget for one column block's CB windows
FAST_TILIZE_MAX_WIDTH_TILES = 255  # fast-tilize LLK legality: block width < 256
LOW_L1_BLOCK_WIDTH_TILES = 8  # low_l1: dimension-independent block width
# Bank-coalesced reads: one wide read per DRAM bank per tile-row (whole-stick blocks only).
COALESCE_BANK_READS = True
# Coalescing trades 32 small DRAM reads for num_banks wide reads + 32 local copies per tile-row;
# it pays under grid-wide DRAM contention (BH: 110 cores 2x faster) and loses ~0.15us on a
# handful of cores (BH: 4-8 cores). Enable only when cores_used >= this * num_banks.
COALESCE_MIN_CORES_PER_BANK = 2
# Rotate each core's bank walk so the grid's requests spread over all banks at once.
BANK_STAGGER = True
# Staging slot budget for the coalesced reader (two slots); sets staging_tile_rows per chunk.
STAGING_SLOT_BUDGET_BYTES = 16 * 1024
# block_height_tiles: None => the core's whole row assignment (Phase 0: one row block).
BLOCK_HEIGHT_TILES = None
# interleaved_2d_split: also cut the columns across cores when the row axis cannot fill the grid.
ENABLE_2D_SPLIT = True
# Leading (batch) dims the reader can pad independently; rank <= 2 + this.
MAX_LEADING_DIMS = 4
# Fill-row source offset: src = fill_base + (dst % FILL_PHASE_BYTES) keeps every local fill copy
# NoC-alignment congruent (any alignment <= this) and in element phase.
FILL_PHASE_BYTES = 64

TILE_WIDTH = 32


def _align_down(x, unit):
    return (x // unit) * unit


def compute_blocking(*, core_cols_tiles, in_page_bytes, out_page_bytes, in_elem_bytes, low_l1, read_alignment):
    """Column blocking: (block_width_tiles, num_col_blocks, last_block_width_tiles)."""
    per_col_tile_bytes = IN_DEPTH_TILE_ROWS * in_page_bytes + OUT_DEPTH_TILE_ROWS * out_page_bytes
    if low_l1:
        max_bw = LOW_L1_BLOCK_WIDTH_TILES
    else:
        max_bw = min(FAST_TILIZE_MAX_WIDTH_TILES, L1_BLOCK_BUDGET_BYTES // per_col_tile_bytes)
    # Every stick-segment offset (col_start * 32 * elem) must stay read-aligned.
    seg_unit_bytes = TILE_WIDTH * in_elem_bytes
    align_unit_tiles = read_alignment // math.gcd(read_alignment, seg_unit_bytes)
    max_bw = max(align_unit_tiles, _align_down(max_bw, align_unit_tiles))

    block_width_tiles = min(core_cols_tiles, max_bw)
    num_col_blocks = math.ceil(core_cols_tiles / block_width_tiles)
    last_block_width_tiles = core_cols_tiles - (num_col_blocks - 1) * block_width_tiles
    return block_width_tiles, num_col_blocks, last_block_width_tiles


def _split_extent(total, parts, unit=1):
    """Contiguous partition of `total` into `parts` ranges, boundaries on multiples of `unit`.

    Returns [(start, count)] with counts ceil/floor over units (the remainder spread over the
    leading ranges, as split_work_to_cores does); the last range absorbs a partial unit.
    """
    units = math.ceil(total / unit)
    base, extra = divmod(units, parts)
    ranges, start = [], 0
    for k in range(parts):
        n_units = base + (1 if k < extra else 0)
        count = min(n_units * unit, total - start)
        ranges.append((start, count))
        start += count
    assert start == total
    return ranges


def _choose_split(num_cores, rows, col_units):
    """(cores_r, cores_c): the regime selector of op_design.md.

    interleaved_row_split (cores_c = 1, cores_r = min(R, grid)) unless the row axis cannot fill the
    grid (R < num_cores); then interleaved_2d_split picks cores_r x cores_c <= grid minimizing the
    busiest core's tile count, ties -> more row cores (longer stick segments), then fewer cores.
    """
    best = (math.ceil(rows / min(rows, num_cores)) * col_units, -min(rows, num_cores), 1)
    best_rc = (min(rows, num_cores), 1)
    if rows >= num_cores or not ENABLE_2D_SPLIT:
        return best_rc
    for cores_r in range(1, rows + 1):
        for cores_c in range(1, min(col_units, num_cores // cores_r) + 1):
            key = (math.ceil(rows / cores_r) * math.ceil(col_units / cores_c), -cores_r, cores_c)
            if key < best:
                best, best_rc = key, (cores_r, cores_c)
    return best_rc


def _rect_assignment(device, tensor_rows_tiles, tensor_cols_tiles, col_unit_tiles):
    """Per-core rectangles: [(core, row_start, rows, col_start, cols)], row-wise core order."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    col_units = math.ceil(tensor_cols_tiles / col_unit_tiles)
    cores_r, cores_c = _choose_split(num_cores, tensor_rows_tiles, col_units)
    # The first cores_r * cores_c cores of the grid, row-wise (one unit each).
    _, all_cores, _, _, _, _ = ttnn.split_work_to_cores(grid, cores_r * cores_c, row_wise=True)
    cores = ttnn.corerange_to_cores(all_cores, None, True)
    row_ranges = _split_extent(tensor_rows_tiles, cores_r)
    col_ranges = _split_extent(tensor_cols_tiles, cores_c, col_unit_tiles)
    assignment = []
    for ri, (row_start, rows) in enumerate(row_ranges):
        for ci, (col_start, cols) in enumerate(col_ranges):
            assignment.append((cores[ri * cores_c + ci], row_start, rows, col_start, cols))
    return all_cores, assignment, cores_c


def _cb(index, num_pages, page_bytes, dtype, cores, tile=None):
    fmt_kwargs = dict(buffer_index=index, data_format=dtype, page_size=page_bytes)
    if tile is not None:
        fmt_kwargs["tile"] = ttnn.TileDescriptor(tile)
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(**fmt_kwargs)],
    )


def create_program_descriptor(input_tensor, output_tensor, *, low_l1=False, padded_shape=None, fill_word=0):
    """`padded_shape`: the output's padded shape (rank >= 2), None = no padding. The input is
    viewed left-expanded with 1s to the same rank; everything past it is filled with `fill_word`
    (the fill encoded in the input format, replicated to 32 bits)."""
    device = input_tensor.device()
    in_shape = [int(d) for d in input_tensor.shape]

    tile_h, tile_w = output_tensor.tile.tile_shape
    assert tile_w == TILE_WIDTH

    out_shape = list(padded_shape) if padded_shape is not None else list(in_shape)
    assert len(out_shape) >= 2 and len(out_shape) >= len(in_shape)
    in_view = [1] * (len(out_shape) - len(in_shape)) + in_shape
    has_padding = in_view != out_shape
    out_lead, in_lead = out_shape[:-2], in_view[:-2]
    assert len(out_lead) <= MAX_LEADING_DIMS, f"tilize: rank {len(out_shape)} > {MAX_LEADING_DIMS + 2}"
    out_lead = [1] * (MAX_LEADING_DIMS - len(out_lead)) + out_lead
    in_lead = [1] * (MAX_LEADING_DIMS - len(in_lead)) + in_lead
    leading_pad = out_lead != in_lead

    # --- Work geometry (output = padded shape; per-image ceil, then fold leading dims) --------
    in_image_height_sticks = in_view[-2]
    tile_rows_per_image = math.ceil(out_shape[-2] / tile_h)
    tensor_rows_tiles = math.prod(out_shape[:-2]) * tile_rows_per_image
    tensor_cols_tiles = math.ceil(out_shape[-1] / TILE_WIDTH)

    # --- Page sizes ---------------------------------------------------------
    in_elem_bytes = input_tensor.element_size()
    in_page_bytes = tile_h * TILE_WIDTH * in_elem_bytes  # one tile's worth of RM data
    out_page_bytes = output_tensor.buffer_page_size()  # one output tile
    stick_page_bytes = input_tensor.buffer_aligned_page_size()
    in_row_bytes = in_view[-1] * in_elem_bytes  # logical data bytes of one input stick

    # --- Per-core rectangles --------------------------------------------------
    read_alignment = (
        ttnn.get_dram_alignment()
        if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        else ttnn.get_l1_alignment()
    )
    # Every stick-segment offset (col_start * 32 * elem) must stay read-aligned.
    seg_unit_bytes = TILE_WIDTH * in_elem_bytes
    align_unit_tiles = read_alignment // math.gcd(read_alignment, seg_unit_bytes)
    all_cores, assignment, cores_c = _rect_assignment(device, tensor_rows_tiles, tensor_cols_tiles, align_unit_tiles)

    def blocking_for(core_cols_tiles):
        return compute_blocking(
            core_cols_tiles=core_cols_tiles,
            in_page_bytes=in_page_bytes,
            out_page_bytes=out_page_bytes,
            in_elem_bytes=in_elem_bytes,
            low_l1=low_l1,
            read_alignment=read_alignment,
        )

    # Column groups: cores whose rectangles have the same column extent share one column walk
    # (CT block widths), so they share one kernel/CB set. At most two (cores_c > 1, ragged last).
    groups = {}  # core_cols_tiles -> list of assignment entries
    for entry in assignment:
        groups.setdefault(entry[4], []).append(entry)

    # Bank-coalesced reads need the block to be the whole stick (contiguous in its bank), an
    # interleaved source (round-robin page -> bank) and no padding (every stick fully present).
    in_mem = input_tensor.memory_config()
    num_banks = ttnn.get_memory_view(device, in_mem.buffer_type).num_banks
    _, whole_width_blocks, _ = blocking_for(tensor_cols_tiles)
    coalesce = (
        COALESCE_BANK_READS
        and cores_c == 1
        and whole_width_blocks == 1
        and not has_padding
        and in_mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and tile_h > 1
        and in_image_height_sticks % tile_h == 0
        and len(assignment) >= COALESCE_MIN_CORES_PER_BANK * num_banks
    )
    max_core_rows = max(entry[2] for entry in assignment)
    staging_tile_rows = max(1, min(max_core_rows, STAGING_SLOT_BUDGET_BYTES // (tile_h * stick_page_bytes)))
    staging_slot_bytes = num_banks * math.ceil(staging_tile_rows * tile_h / num_banks) * stick_page_bytes
    staging_bytes = 2 * staging_slot_bytes
    # Bank stagger (coalesced reader): core k starts its bank walk at bank k % groups, where
    # groups = min(num_banks, sticks in the smallest chunk) so the offset is always in range.
    min_chunk_sticks = min(min(entry[2] for entry in assignment), staging_tile_rows) * tile_h
    stagger_groups = min(num_banks, min_chunk_sticks)
    core_index = {(entry[0].x, entry[0].y): k for k, entry in enumerate(assignment)}

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    out_tile = ttnn.Tile([tile_h, TILE_WIDTH])
    kernels, cbs = [], []
    for core_cols_tiles, entries in groups.items():
        group_cores = (
            all_cores if len(groups) == 1 else ttnn.CoreRangeSet([ttnn.CoreRange(e[0], e[0]) for e in entries])
        )
        block_width_tiles, num_col_blocks, last_block_width_tiles = blocking_for(core_cols_tiles)
        has_tail = last_block_width_tiles != block_width_tiles

        # --- Circular buffers (per group: ring quanta = this group's block widths) ----------
        cbs.append(
            _cb(
                CB_RM_STICKS,
                IN_DEPTH_TILE_ROWS * block_width_tiles,
                in_page_bytes,
                input_tensor.dtype,
                group_cores,
                out_tile,
            )
        )
        cbs.append(
            _cb(
                CB_TILES_OUT,
                OUT_DEPTH_TILE_ROWS * block_width_tiles,
                out_page_bytes,
                output_tensor.dtype,
                group_cores,
                out_tile,
            )
        )
        if coalesce:
            cbs.append(_cb(CB_STICK_STAGING, 1, staging_bytes, input_tensor.dtype, group_cores))
        fill_row_bytes = block_width_tiles * seg_unit_bytes + FILL_PHASE_BYTES
        if has_padding:
            # + FILL_PHASE_BYTES so the kernel can align the row base up to FILL_PHASE_BYTES.
            cbs.append(_cb(CB_PAD_FILL, 1, fill_row_bytes + FILL_PHASE_BYTES, input_tensor.dtype, group_cores))
        if has_tail:
            # The ragged last column block gets its own ring so every ring is filled in
            # one quantum (no contiguous block can straddle the ring end).
            cbs.append(
                _cb(
                    CB_RM_STICKS_TAIL,
                    IN_DEPTH_TILE_ROWS * last_block_width_tiles,
                    in_page_bytes,
                    input_tensor.dtype,
                    group_cores,
                    out_tile,
                )
            )
            cbs.append(
                _cb(
                    CB_TILES_OUT_TAIL,
                    OUT_DEPTH_TILE_ROWS * last_block_width_tiles,
                    out_page_bytes,
                    output_tensor.dtype,
                    group_cores,
                    out_tile,
                )
            )

        # --- Kernel args -----------------------------------------------------------------
        col_geometry = [block_width_tiles, last_block_width_tiles, num_col_blocks, int(has_tail)]
        reader_ct = [
            CB_RM_STICKS,
            CB_RM_STICKS_TAIL,
            tile_h,
            seg_unit_bytes,  # bytes of one tile-column of one stick
            in_image_height_sticks,
            tile_rows_per_image,
            stick_page_bytes,
            *col_geometry,
            int(coalesce),
            num_banks,
            CB_STICK_STAGING,
            staging_tile_rows,
            int(has_padding),
            CB_PAD_FILL,
            in_row_bytes,
            fill_row_bytes,
            FILL_PHASE_BYTES,
            int(leading_pad),
            *out_lead,
            *in_lead,
        ]
        reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        writer_ct = [CB_TILES_OUT, CB_TILES_OUT_TAIL, tensor_cols_tiles, out_page_bytes, *col_geometry]
        writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        compute_ct = [CB_RM_STICKS, CB_RM_STICKS_TAIL, CB_TILES_OUT, CB_TILES_OUT_TAIL, *col_geometry]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, row_start_tile, core_rows_tiles, col_start_tile, _ in entries:
            block_height_tiles = (
                core_rows_tiles if BLOCK_HEIGHT_TILES is None else min(BLOCK_HEIGHT_TILES, core_rows_tiles)
            )
            rect = [row_start_tile, core_rows_tiles, col_start_tile, block_height_tiles]
            stagger = (core_index[(core.x, core.y)] % stagger_groups) if (BANK_STAGGER and coalesce) else 0
            reader_rt[core.x][core.y] = [in_addr, *rect, stagger, fill_word]
            writer_rt[core.x][core.y] = [out_addr, *rect]
            compute_rt[core.x][core.y] = [core_rows_tiles, block_height_tiles]

        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
                core_ranges=group_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
                core_ranges=group_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
                core_ranges=group_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
