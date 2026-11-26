# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout program descriptor — regime R1 `interleaved_untilize_2d` (+ R1a live-face reads).

Single source of truth for every block knob is `plan_blocking()`; the CB page counts, loop trip
counts and the core grid are all derived from its result (op_design.md, Blocking Model).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Blocking-model knobs (host constants, op_design.md "Knobs that are not per-axis") ----
BLOCK_TILES_TARGET = 32  # tiles per block (amortizes per-block barrier / flush / CB handshake)
DEPTH_IN = 2  # cb_input_tiles depth in blocks (double buffering)
DEPTH_OUT = 2  # cb_output_rows depth in blocks (double buffering)
L1_OP_BOUND = 1347776  # spec bound on per-core L1 for this op (BB-05)
DEST_TILES = 8  # pack_untilize capacity, 16-bit DEST half-sync (pack_untilize.h:102)

TILE_HW = 32
FACE_HW = 16

# CB slots (semantic names)
CB_INPUT_TILES = 0  # reader -> compute, TILE pages (opaque UInt16)
CB_OUTPUT_ROWS = 1  # compute -> writer, row-major rows of a unit in tile-sized pages (opaque UInt16)

_PLAN_CACHE: dict = {}


def _cdiv(a, b):
    return -(-a // b)


def _align_up(a, b):
    return _cdiv(a, b) * b


def _floor_dest(x):
    return x if x <= DEST_TILES else DEST_TILES * (x // DEST_TILES)


def _num_l1_banks(device):
    try:
        return ttnn.device.GetMemoryView(device, ttnn.BufferType.L1).num_banks
    except Exception:  # noqa: BLE001 — conservative fallback: one bank per worker core
        grid = device.compute_with_storage_grid_size()
        return grid.x * grid.y


def _grid_key(grid):
    return (grid.x, grid.y)


def plan_blocking(device, shape, out_buffer_type, *, tile_bytes=2048, elem_bytes=2):
    """The R1 work split (op_design.md, Work-split formula). Memoized per key (BB-07)."""
    grid = device.compute_with_storage_grid_size()
    key = (tuple(shape), out_buffer_type, _grid_key(grid), tile_bytes, elem_bytes)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    H, W = shape[-2], shape[-1]
    N = 1
    for d in shape[:-2]:
        N *= d
    tiles_per_image_h = _cdiv(H, TILE_HW)
    tensor_tile_rows = N * tiles_per_image_h
    tensor_tile_cols = _cdiv(W, TILE_HW)
    num_cores = grid.x * grid.y

    if out_buffer_type == ttnn.BufferType.L1:
        num_out_pages = N * H
        out_aligned_page_bytes = _align_up(W * elem_bytes, ttnn.get_l1_alignment())
        out_l1_bytes_per_core = _cdiv(num_out_pages, _num_l1_banks(device)) * out_aligned_page_bytes
    else:
        out_l1_bytes_per_core = 0
    op_l1_budget = L1_OP_BOUND - out_l1_bytes_per_core
    depth_sum = DEPTH_IN + DEPTH_OUT
    max_block_cols = _floor_dest(op_l1_budget // (depth_sum * tile_bytes))
    assert max_block_cols >= 1, "g6_to_layout: no L1 left for one tile per buffer"

    occupancy_chunks = _cdiv(num_cores, tensor_tile_rows) if tensor_tile_rows < num_cores else 1
    chunks0 = max(_cdiv(tensor_tile_cols, max_block_cols), min(tensor_tile_cols, occupancy_chunks))
    block_tile_cols = _cdiv(tensor_tile_cols, chunks0)
    if block_tile_cols > DEST_TILES:
        block_tile_cols = min(_align_up(block_tile_cols, DEST_TILES), max_block_cols)
    num_col_chunks = _cdiv(tensor_tile_cols, block_tile_cols)
    max_rows_fit = op_l1_budget // (depth_sum * block_tile_cols * tile_bytes)
    block_tile_rows = max(1, min(BLOCK_TILES_TARGET // block_tile_cols, max_rows_fit))
    num_units = tensor_tile_rows * num_col_chunks

    cb_bytes = depth_sum * block_tile_rows * block_tile_cols * tile_bytes
    assert cb_bytes <= op_l1_budget, f"g6_to_layout: CB total {cb_bytes} B exceeds op L1 budget {op_l1_budget} B"

    (_, all_cores, core_group_1, core_group_2, units_g1, units_g2) = ttnn.split_work_to_cores(
        grid, num_units, row_wise=True
    )
    assignment = []  # (core, start_unit, units_this_core)
    start = 0
    for group, per_core in ((core_group_1, units_g1), (core_group_2, units_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_units

    plan = {
        "regime": "interleaved_untilize_2d",
        "N": N,
        "H": H,
        "W": W,
        "tiles_per_image_h": tiles_per_image_h,
        "tensor_tile_rows": tensor_tile_rows,
        "tensor_tile_cols": tensor_tile_cols,
        "tile_bytes": tile_bytes,
        "elem_bytes": elem_bytes,
        "op_l1_budget": op_l1_budget,
        "max_block_cols": max_block_cols,
        "block_tile_cols": block_tile_cols,
        "block_tile_rows": block_tile_rows,
        "num_col_chunks": num_col_chunks,
        "num_units": num_units,
        "depth_in": DEPTH_IN,
        "depth_out": DEPTH_OUT,
        "w_tail_half": 1 if 1 <= W % TILE_HW <= FACE_HW else 0,
        "h_tail_half": 1 if 1 <= H % TILE_HW <= FACE_HW else 0,
        "all_cores": all_cores,
        "assignment": assignment,
        "num_cores_used": len(assignment),
    }
    _PLAN_CACHE[key] = plan
    return plan


def create_program_descriptor(input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    shape = tuple(input_tensor.shape)
    out_buffer_type = output_tensor.memory_config().buffer_type
    plan = plan_blocking(device, shape, out_buffer_type)

    all_cores = plan["all_cores"]
    btc = plan["block_tile_cols"]
    btr = plan["block_tile_rows"]
    tile_bytes = plan["tile_bytes"]

    # ---- CBs: opaque 16-bit (UInt16) so bfloat16 bits pass the engines unchanged ----
    cb_input_tiles = ttnn.CBDescriptor(
        total_size=DEPTH_IN * btr * btc * tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_INPUT_TILES, data_format=ttnn.uint16, page_size=tile_bytes)
        ],
    )
    cb_output_rows = ttnn.CBDescriptor(
        total_size=DEPTH_OUT * btr * btc * tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_ROWS, data_format=ttnn.uint16, page_size=tile_bytes)
        ],
    )

    # ---- Reader ----
    reader_ct = [
        CB_INPUT_TILES,
        btc,
        btr,
        plan["tensor_tile_cols"],
        plan["num_col_chunks"],
        plan["tiles_per_image_h"],
        tile_bytes,
        plan["w_tail_half"],
        plan["h_tail_half"],
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---- Writer ----
    writer_ct = [
        CB_OUTPUT_ROWS,
        btc,
        btr,
        plan["num_col_chunks"],
        plan["tiles_per_image_h"],
        plan["H"],
        plan["W"] * plan["elem_bytes"],  # bytes per logical output row
        tile_bytes,
        ttnn.get_l1_alignment() if out_buffer_type == ttnn.BufferType.L1 else 16,  # write length granule
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, start_unit, units in plan["assignment"]:
        reader_rt[core.x][core.y] = [in_addr, start_unit, units]
        writer_rt[core.x][core.y] = [out_addr, start_unit, units]
        compute_rt[core.x][core.y] = [units]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[CB_INPUT_TILES, CB_OUTPUT_ROWS, btc],
        runtime_args=compute_rt,
        config=compute_config,
    )
    return ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[],
        cbs=[cb_input_tiles, cb_output_rows],
    )
