# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast — ProgramDescriptor for regime R1 `tile_stream` (op_design.md, Blocking Model).

Flat `tile` axis (batch x tile_row x tile_col folded, contiguous in page order) split into contiguous
ranges over the full compute grid (`split_work_to_cores(row_wise=True)`). Each core walks its range in
blocks of `block_tiles` tiles: one reader barrier, one CB handshake per kernel per block.

Knobs (single source each — every dependent quantity derives from them):
  BUFFER_DEPTH     — CB depth knob; effective depth = min(BUFFER_DEPTH, max blocks per core)
  BLOCK_TILES_CAP  — optional upper bound on block_tiles (None = coarsest block that fits L1)
  L1_OP_BUDGET_BYTES / L1_SLACK_BYTES — L1 grant (BB-05) and allocator headroom
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- knobs ----------------------------------------------------------------------------------------
BUFFER_DEPTH = 2
BLOCK_TILES_CAP = None  # None: block = whole per-core range clamped to the L1 grant
L1_OP_BUDGET_BYTES = 1347776
L1_SLACK_BYTES = 65536

# ---- CB indices (semantic names) ------------------------------------------------------------------
CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 1

TILE_HW = 32

# Runtime-arg slots patched on a program-cache hit (common runtime args, one per dataflow kernel).
_READER_KERNEL_IDX = 0
_WRITER_KERNEL_IDX = 1


def _div_up(a, b):
    return (a + b - 1) // b


def _tensor_tiles(shape):
    """Tile count of a TILE tensor: prod(batch dims) * ceil(H/32) * ceil(W/32)."""
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    batch = 1
    for d in shape[:-2]:
        batch *= d
    return batch * _div_up(h, TILE_HW) * _div_up(w, TILE_HW)


def _num_l1_banks(device):
    try:
        return device.num_banks(ttnn.BufferType.L1)
    except Exception:  # noqa: BLE001 — conservative fallback: one bank per worker core
        g = device.compute_with_storage_grid_size()
        return g.x * g.y


def plan_blocks(core_tiles_max, in_page_bytes, out_page_bytes, out_l1_bytes_per_bank):
    """Return (block_tiles, effective_depth): the coarsest block that fits the L1 grant."""
    cb_budget = L1_OP_BUDGET_BYTES - L1_SLACK_BYTES - out_l1_bytes_per_bank
    pair_bytes = in_page_bytes + out_page_bytes
    block_tiles = core_tiles_max
    if BLOCK_TILES_CAP is not None:
        block_tiles = min(block_tiles, BLOCK_TILES_CAP)
    # Depth depends on the number of blocks, which depends on the block; iterate to a fixpoint.
    while True:
        depth = min(BUFFER_DEPTH, _div_up(core_tiles_max, block_tiles))
        cap = cb_budget // (depth * pair_bytes)
        assert cap >= 1, f"g6_typecast: L1 grant too small for one block ({cb_budget} B)"
        if block_tiles <= cap:
            return block_tiles, depth
        block_tiles = cap


def create_program_descriptor(input_tensor, output_tensor):
    device = input_tensor.device()
    shape = tuple(input_tensor.padded_shape)
    logical_shape = tuple(input_tensor.shape)
    out_dtype = output_tensor.dtype

    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    tensor_tiles = _tensor_tiles(logical_shape)
    assert tensor_tiles == input_tensor.buffer_num_pages() == output_tensor.buffer_num_pages(), (
        tensor_tiles,
        input_tensor.buffer_num_pages(),
        output_tensor.buffer_num_pages(),
    )

    # ---- work distribution: contiguous tile ranges over the full grid, row-wise ----
    grid = device.compute_with_storage_grid_size()
    (num_cores, all_cores, core_group_1, core_group_2, tiles_g1, tiles_g2) = ttnn.split_work_to_cores(
        grid, tensor_tiles, row_wise=True
    )
    core_tiles_max = tiles_g1

    # ---- block plan (L1 grant includes the op's own L1 output share) ----
    out_l1_bytes_per_bank = 0
    if output_tensor.memory_config().buffer_type == ttnn.BufferType.L1:
        out_l1_bytes_per_bank = _div_up(tensor_tiles, _num_l1_banks(device)) * out_page_bytes
    block_tiles, effective_depth = plan_blocks(core_tiles_max, in_page_bytes, out_page_bytes, out_l1_bytes_per_bank)
    cb_pages = effective_depth * block_tiles

    # ---- bf8b padding-column masking (shared exponent must not see padding values) ----
    w = logical_shape[-1]
    w_tail = w % TILE_HW
    tensor_tile_cols = _div_up(w, TILE_HW)
    mask_pad_columns = int(out_dtype == ttnn.bfloat8_b and w % 16 != 0)

    # ---- CBs ----
    cb_input_tiles = ttnn.CBDescriptor(
        total_size=cb_pages * in_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page_bytes
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=cb_pages * out_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUTPUT_TILES, data_format=out_dtype, page_size=out_page_bytes)
        ],
    )

    # ---- per-core RT args: [start_tile, core_tiles] ----
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start_tile = 0
    for group, count in ((core_group_1, tiles_g1), (core_group_2, tiles_g2)):
        if count == 0 or group.num_cores() == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            reader_rt[core.x][core.y] = [start_tile, count]
            writer_rt[core.x][core.y] = [start_tile, count]
            compute_rt[core.x][core.y] = [count]
            start_tile += count
    assert start_tile == tensor_tiles

    # ---- kernels ----
    reader_ct = [CB_INPUT_TILES, block_tiles, in_page_bytes, mask_pad_columns, tensor_tile_cols, w_tail]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct = [CB_OUTPUT_TILES, block_tiles, out_page_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    fp32_dest = out_dtype in (ttnn.float32, ttnn.int32, ttnn.uint32)
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = fp32_dest
    compute_config.bfp8_pack_precise = out_dtype == ttnn.bfloat8_b
    compute_config.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[CB_INPUT_TILES, CB_OUTPUT_TILES, block_tiles],
        runtime_args=compute_rt,
        config=compute_config,
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],  # order matches _READER_KERNEL_IDX / _WRITER_KERNEL_IDX
        semaphores=[],
        cbs=[cb_input_tiles, cb_output_tiles],
    )
    plan = {
        "regime": "tile_stream",
        "num_cores": num_cores,
        "tensor_tiles": tensor_tiles,
        "core_tiles_max": core_tiles_max,
        "block_tiles": block_tiles,
        "effective_depth": effective_depth,
        "mask_pad_columns": bool(mask_pad_columns),
    }
    return program_descriptor, plan


def patch_buffer_addresses(program_descriptor, input_tensor, output_tensor):
    """Program-cache hit: only the buffer addresses change (BB-07)."""
    program_descriptor.kernels[_READER_KERNEL_IDX].common_runtime_args = [input_tensor.buffer_address()]
    program_descriptor.kernels[_WRITER_KERNEL_IDX].common_runtime_args = [output_tensor.buffer_address()]
