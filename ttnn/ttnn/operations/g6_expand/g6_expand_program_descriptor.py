# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand — program descriptor (regime R1 of op_design.md).

R1: 2D grid split of out over (image group, row group, column group) blocks;
every core reads row 0 of its x tiles (one partial 576 B DRAM read per tile),
replicates it in place into a broadcast tile B(n, w) with self-aimed L1->L1
NoC reads, and writes that one B tile to all out pages of its row group.

Every block knob is defined ONCE here (host constants / host-derived values)
and every CB size, trip count and core count derives from it:
  BLOCK_IMAGES          — extent of the image axis per block (Phase 0 = 1)
  MIN_BLOCK_TILE_ROWS   — lower bound of the out-tile-row extent per block (Phase 0 = 2);
                          block_tile_rows is derived from it and the grid
  BCAST_BUFFER_DEPTH    — cb_bcast_row_tiles depth in blocks (Phase 0 = 2)
  block_tile_cols       — derived: ceil(Wt / num_cores), clamped by the L1 budget
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- knobs (single source of truth) ---------------------------------------
BLOCK_IMAGES = 1
MIN_BLOCK_TILE_ROWS = 2
BCAST_BUFFER_DEPTH = 2

# ---- L1 budget (BB-05) -----------------------------------------------------
L1_BYTES_LEFT_BY_MODEL = 1347776
RESERVED_MARGIN = 4096
L1_BUDGET_FOR_OP = L1_BYTES_LEFT_BY_MODEL - RESERVED_MARGIN

# ---- CB slot ---------------------------------------------------------------
CB_BCAST_ROW_TILES = 0

TILE_HW = 32


def _div_up(a, b):
    return -(-a // b)


def _round_up(a, m):
    return _div_up(a, m) * m


def compute_plan(
    input_shape,
    out_shape,
    grid_x,
    grid_y,
    tile_bytes,
    element_size,
    *,
    block_images=BLOCK_IMAGES,
    min_block_tile_rows=MIN_BLOCK_TILE_ROWS,
    bcast_buffer_depth=BCAST_BUFFER_DEPTH,
):
    """Pure host derivation of the blocking plan (Work Distribution table of op_design.md)."""
    num_cores = grid_x * grid_y
    tensor_images = int(out_shape[0]) * int(out_shape[1])
    tensor_tile_cols = _div_up(int(out_shape[3]), TILE_HW)  # Wt
    tensor_out_tile_rows = _div_up(int(out_shape[2]), TILE_HW)  # Ht (per image)

    block_images = max(1, min(block_images, tensor_images))
    image_groups = _div_up(tensor_images, block_images)

    max_block_tile_cols = max(1, L1_BUDGET_FOR_OP // (bcast_buffer_depth * block_images * tile_bytes))
    block_tile_cols = min(max(1, _div_up(tensor_tile_cols, num_cores)), max_block_tile_cols)
    col_groups = _div_up(tensor_tile_cols, block_tile_cols)

    row_groups_fill = max(1, num_cores // (image_groups * col_groups))
    row_groups_cap = _div_up(tensor_out_tile_rows, max(1, min_block_tile_rows))
    row_groups = min(max(min(row_groups_fill, row_groups_cap), 1), tensor_out_tile_rows)
    block_tile_rows = _div_up(tensor_out_tile_rows, row_groups)
    row_groups = _div_up(tensor_out_tile_rows, block_tile_rows)

    num_blocks = image_groups * row_groups * col_groups

    # Partial x read: row 0 of faces 0 and 1 (face 0 whole + face-1 row 0), DRAM-aligned.
    face_hw = TILE_HW // 2
    face_row_bytes = face_hw * element_size
    face_bytes = face_hw * face_row_bytes
    x_row_read_bytes = _round_up(face_bytes + face_row_bytes, ttnn.get_dram_alignment())
    if x_row_read_bytes > tile_bytes:
        x_row_read_bytes = tile_bytes

    # Per-core block ranges: contiguous block ids, row-wise core order (== split_work_to_cores row_wise).
    num_cores_used = min(num_blocks, num_cores)
    base, extra = divmod(num_blocks, num_cores_used)
    core_blocks = []  # (CoreCoord-xy, first_block, num_blocks_this_core)
    next_block = 0
    for k in range(num_cores_used):
        n_this = base + (1 if k < extra else 0)
        core_blocks.append(((k % grid_x, k // grid_x), next_block, n_this))
        next_block += n_this
    assert next_block == num_blocks

    return {
        "grid": (grid_x, grid_y),
        "num_cores": num_cores,
        "num_cores_used": num_cores_used,
        "tensor_images": tensor_images,
        "tensor_tile_cols": tensor_tile_cols,
        "tensor_out_tile_rows": tensor_out_tile_rows,
        "block_images": block_images,
        "image_groups": image_groups,
        "block_tile_cols": block_tile_cols,
        "max_block_tile_cols": max_block_tile_cols,
        "col_groups": col_groups,
        "block_tile_rows": block_tile_rows,
        "row_groups": row_groups,
        "num_blocks": num_blocks,
        "bcast_buffer_depth": bcast_buffer_depth,
        "min_block_tile_rows": min_block_tile_rows,
        "tile_bytes": tile_bytes,
        "face_row_bytes": face_row_bytes,
        "face_bytes": face_bytes,
        "face_rows": face_hw,
        "x_row_read_bytes": x_row_read_bytes,
        "core_blocks": core_blocks,
    }


def _core_ranges(num_cores_used, grid_x):
    """First num_cores_used cores in row-wise order: <= 2 rectangles (BB-17)."""
    full_rows, rem = divmod(num_cores_used, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


# Address RT arg index (both kernels): patched on a cache hit.
RT_ADDR_INDEX = 0


def create_program_descriptor(input_tensor, output_tensor, plan):
    core_set = _core_ranges(plan["num_cores_used"], plan["grid"][0])
    tile_bytes = plan["tile_bytes"]
    block_pages = plan["block_images"] * plan["block_tile_cols"]

    cb_bcast_row_tiles = ttnn.CBDescriptor(
        total_size=block_pages * plan["bcast_buffer_depth"] * tile_bytes,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BCAST_ROW_TILES,
                data_format=input_tensor.dtype,
                page_size=tile_bytes,
            )
        ],
    )

    reader_ct = [
        CB_BCAST_ROW_TILES,
        plan["x_row_read_bytes"],
        plan["face_row_bytes"],
        plan["face_bytes"],
        plan["face_rows"],
        tile_bytes,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct = [CB_BCAST_ROW_TILES, tile_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for (cx, cy), first_block, n_blocks in plan["core_blocks"]:
        reader_rt[cx][cy] = [
            x_addr,
            first_block,
            n_blocks,
            plan["tensor_tile_cols"],
            plan["row_groups"],
            plan["col_groups"],
            plan["block_tile_cols"],
            plan["block_images"],
            plan["tensor_images"],
        ]
        writer_rt[cx][cy] = [
            out_addr,
            first_block,
            n_blocks,
            plan["tensor_tile_cols"],
            plan["tensor_out_tile_rows"],
            plan["row_groups"],
            plan["col_groups"],
            plan["block_tile_cols"],
            plan["block_tile_rows"],
            plan["block_images"],
            plan["tensor_images"],
        ]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    # No compute kernel: data_movement op (design: no arithmetic, bit-exact byte moves only).
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb_bcast_row_tiles])


def patch_addresses(descriptor, plan, input_tensor, output_tensor):
    """Cache hit: rewrite only the buffer-address RT arg of each core (BB-07)."""
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    reader_rt = descriptor.kernels[0].runtime_args
    writer_rt = descriptor.kernels[1].runtime_args
    for (cx, cy), _, _ in plan["core_blocks"]:
        reader_rt[cx][cy][RT_ADDR_INDEX] = x_addr
        writer_rt[cx][cy][RT_ADDR_INDEX] = out_addr
