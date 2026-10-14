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
# block_height_tiles: None => the core's whole row assignment (Phase 0: one row block).
BLOCK_HEIGHT_TILES = None

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


def _row_assignment(device, tensor_rows_tiles):
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(
        grid, tensor_rows_tiles, row_wise=True
    )
    assignment = []  # (core, row_start_tile, core_rows_tiles)
    start = 0
    for group, per_core in ((core_group_1, rows_g1), (core_group_2, rows_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == tensor_rows_tiles
    return all_cores, assignment


def _cb(index, num_pages, page_bytes, dtype, cores, tile=None):
    fmt_kwargs = dict(buffer_index=index, data_format=dtype, page_size=page_bytes)
    if tile is not None:
        fmt_kwargs["tile"] = ttnn.TileDescriptor(tile)
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(**fmt_kwargs)],
    )


def create_program_descriptor(input_tensor, output_tensor, *, low_l1=False):
    device = input_tensor.device()
    shape = list(input_tensor.shape)

    tile_h, tile_w = output_tensor.tile.tile_shape
    assert tile_w == TILE_WIDTH

    # --- Work geometry (per-image ceil, then fold leading dims) --------------
    image_height_sticks = int(shape[-2])
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    tile_rows_per_image = math.ceil(image_height_sticks / tile_h)
    tensor_rows_tiles = leading * tile_rows_per_image
    tensor_cols_tiles = math.ceil(int(shape[-1]) / TILE_WIDTH)

    # --- Page sizes ---------------------------------------------------------
    in_elem_bytes = input_tensor.element_size()
    in_page_bytes = tile_h * TILE_WIDTH * in_elem_bytes  # one tile's worth of RM data
    out_page_bytes = output_tensor.buffer_page_size()  # one output tile
    stick_page_bytes = input_tensor.buffer_aligned_page_size()

    # --- Per-core rectangle --------------------------------------------------
    all_cores, assignment = _row_assignment(device, tensor_rows_tiles)
    col_start_tile = 0
    core_cols_tiles = tensor_cols_tiles  # every core owns all columns in this regime

    read_alignment = (
        ttnn.get_dram_alignment()
        if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        else ttnn.get_l1_alignment()
    )
    block_width_tiles, num_col_blocks, last_block_width_tiles = compute_blocking(
        core_cols_tiles=core_cols_tiles,
        in_page_bytes=in_page_bytes,
        out_page_bytes=out_page_bytes,
        in_elem_bytes=in_elem_bytes,
        low_l1=low_l1,
        read_alignment=read_alignment,
    )
    has_tail = last_block_width_tiles != block_width_tiles

    # Bank-coalesced reads need the block to be the whole stick (contiguous in its bank) and
    # an interleaved source (round-robin page -> bank).
    in_mem = input_tensor.memory_config()
    num_banks = ttnn.get_memory_view(device, in_mem.buffer_type).num_banks
    coalesce = (
        COALESCE_BANK_READS
        and num_col_blocks == 1
        and in_mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and tile_h > 1
    )
    staging_bytes = num_banks * math.ceil(tile_h / num_banks) * stick_page_bytes

    # --- Circular buffers ----------------------------------------------------
    out_tile = ttnn.Tile([tile_h, TILE_WIDTH])
    cbs = [
        _cb(
            CB_RM_STICKS,
            IN_DEPTH_TILE_ROWS * block_width_tiles,
            in_page_bytes,
            input_tensor.dtype,
            all_cores,
            out_tile,
        ),
        _cb(
            CB_TILES_OUT,
            OUT_DEPTH_TILE_ROWS * block_width_tiles,
            out_page_bytes,
            output_tensor.dtype,
            all_cores,
            out_tile,
        ),
    ]
    if coalesce:
        cbs.append(_cb(CB_STICK_STAGING, 1, staging_bytes, input_tensor.dtype, all_cores))
    if has_tail:
        # The ragged last column block gets its own ring so every ring is filled in
        # one quantum (no contiguous block can straddle the ring end).
        cbs.append(
            _cb(
                CB_RM_STICKS_TAIL,
                IN_DEPTH_TILE_ROWS * last_block_width_tiles,
                in_page_bytes,
                input_tensor.dtype,
                all_cores,
                out_tile,
            )
        )
        cbs.append(
            _cb(
                CB_TILES_OUT_TAIL,
                OUT_DEPTH_TILE_ROWS * last_block_width_tiles,
                out_page_bytes,
                output_tensor.dtype,
                all_cores,
                out_tile,
            )
        )

    # --- Kernel args ---------------------------------------------------------
    col_geometry = [block_width_tiles, last_block_width_tiles, num_col_blocks, int(has_tail)]

    reader_ct = [
        CB_RM_STICKS,
        CB_RM_STICKS_TAIL,
        tile_h,
        TILE_WIDTH * in_elem_bytes,  # bytes of one tile-column of one stick
        image_height_sticks,
        tile_rows_per_image,
        stick_page_bytes,
        *col_geometry,
        int(coalesce),
        num_banks,
        CB_STICK_STAGING,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct = [CB_TILES_OUT, CB_TILES_OUT_TAIL, tensor_cols_tiles, out_page_bytes, *col_geometry]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [CB_RM_STICKS, CB_RM_STICKS_TAIL, CB_TILES_OUT, CB_TILES_OUT_TAIL, *col_geometry]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, row_start_tile, core_rows_tiles in assignment:
        block_height_tiles = core_rows_tiles if BLOCK_HEIGHT_TILES is None else min(BLOCK_HEIGHT_TILES, core_rows_tiles)
        rect = [row_start_tile, core_rows_tiles, col_start_tile, block_height_tiles]
        reader_rt[core.x][core.y] = [in_addr, *rect]
        writer_rt[core.x][core.y] = [out_addr, *rect]
        compute_rt[core.x][core.y] = [core_rows_tiles, block_height_tiles]

    import os

    _abl = [(d, "1") for d in os.environ.get("TILIZE_ABLATE", "").split(",") if d]
    reader = ttnn.KernelDescriptor(
        defines=_abl,
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        defines=_abl,
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
