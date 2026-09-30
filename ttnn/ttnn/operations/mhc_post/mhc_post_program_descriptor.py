# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — ProgramDescriptor (regime `flat_stream`, see op_design.md Blocking Model).

Flattened (token-tile row r, column tile c) units, r-major, split contiguously over the full compute
grid (`split_work_to_cores(..., row_wise=True)`). Per core: segments (one token row each), blocks of
`block_col_tiles` columns. Every block knob below is defined once; CB sizes, CT/RT args derive from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HW = 32

# ---- CB slots (semantic names) ----
CB_SUBLAYER_TILES = 0  # F block tiles              reader  -> compute
CB_RESIDUAL_TILES = 1  # X block tiles (n streams)  reader  -> compute
CB_COEF_RAW = 2  # raw post / comb tiles of one token row, reader-private scratch
CB_COEF_BCAST = 3  # n + n^2 column-broadcast fp32 coefficient tiles, reader -> compute
CB_OUTPUT_TILES = 16  # X' block tiles (n streams) compute -> writer

# ---- Block-model knobs (single source of truth) ----
BLOCK_TOKEN_TILES = 1  # a block never spans two token rows (realized by segments)
DEPTH_IN = 2  # blocks in flight on cb_sublayer_tiles / cb_residual_tiles
DEPTH_OUT = 2  # blocks in flight on cb_output_tiles
COEF_DEPTH = 2  # token-row coefficient sets in flight on cb_coef_bcast
L1_BUDGET_BYTES = 1 << 20  # CB budget per core
NUM_CIRCULAR_BUFFERS = 64  # length of ComputeConfigDescriptor.unpack_to_dest_mode

assert BLOCK_TOKEN_TILES == 1, "flat_stream realizes block_token_tiles through segments; only 1 is built"


def _tensor_token_tiles(shape) -> int:
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * math.ceil(shape[-2] / TILE_HW)


def _work_assignment(grid_size, total_units):
    """Contiguous flattened split: [(core, start_unit, num_units), ...] in split order."""
    (
        _,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_units, row_wise=True)
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_per_core_g1), (core_group_2, units_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == total_units
    return all_cores, assignment


def _max_segment_col_tiles(assignment, col_tiles_per_row) -> int:
    longest = 0
    for _, start, count in assignment:
        u, left = start, count
        while left:
            seg = min(col_tiles_per_row - u % col_tiles_per_row, left)
            longest = max(longest, seg)
            u += seg
            left -= seg
    return longest


def _block_col_tiles_fit(n, sublayer_tile_bytes, residual_tile_bytes, coef_tile_bytes, num_raw_tiles) -> int:
    """Closed form from l1_ledger.md: coarsest block that fits L1_BUDGET_BYTES."""
    coef_bytes = COEF_DEPTH * (n + n * n) * coef_tile_bytes + num_raw_tiles * coef_tile_bytes
    per_col = DEPTH_IN * (sublayer_tile_bytes + n * residual_tile_bytes) + DEPTH_OUT * n * residual_tile_bytes
    return (L1_BUDGET_BYTES - coef_bytes) // per_col


def _cb(index, dtype, page_bytes, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    residual: ttnn.Tensor,
    post: ttnn.Tensor,
    comb: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()

    n = post.shape[-1]
    col_tiles_per_row = input_tensor.shape[-1] // TILE_HW  # Ct
    token_tiles = _tensor_token_tiles(list(input_tensor.padded_shape))
    total_units = token_tiles * col_tiles_per_row
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    num_raw_tiles = post_tiles_per_row + comb_tiles_per_row
    num_coef_tiles = n + n * n

    sublayer_page = input_tensor.buffer_page_size()
    residual_page = residual.buffer_page_size()
    output_page = output_tensor.buffer_page_size()
    coef_page = post.buffer_page_size()
    assert comb.buffer_page_size() == coef_page
    assert output_page == residual_page

    # ---- work split + block size ----
    grid_size = device.compute_with_storage_grid_size()
    all_cores, assignment = _work_assignment(grid_size, total_units)
    block_col_tiles_fit = _block_col_tiles_fit(n, sublayer_page, residual_page, coef_page, num_raw_tiles)
    assert block_col_tiles_fit >= 1, "mhc_post: coefficient set + one column block does not fit L1_BUDGET_BYTES"
    block_col_tiles = min(block_col_tiles_fit, _max_segment_col_tiles(assignment, col_tiles_per_row))

    # ---- circular buffers ----
    cbs = [
        _cb(CB_SUBLAYER_TILES, input_tensor.dtype, sublayer_page, DEPTH_IN * block_col_tiles, all_cores),
        _cb(CB_RESIDUAL_TILES, residual.dtype, residual_page, DEPTH_IN * n * block_col_tiles, all_cores),
        _cb(CB_COEF_RAW, post.dtype, coef_page, num_raw_tiles, all_cores),
        _cb(CB_COEF_BCAST, post.dtype, coef_page, COEF_DEPTH * num_coef_tiles, all_cores),
        _cb(CB_OUTPUT_TILES, output_tensor.dtype, output_page, DEPTH_OUT * n * block_col_tiles, all_cores),
    ]

    # ---- reader ----
    reader_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        post_tiles_per_row,
        comb_tiles_per_row,
        sublayer_page,
        residual_page,
        coef_page,
        TILE_HW,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_RAW,
        CB_COEF_BCAST,
    ]
    for t in (input_tensor, residual, post, comb):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    # ---- writer ----
    writer_ct = [n, col_tiles_per_row, block_col_tiles, output_page, CB_OUTPUT_TILES]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- compute ----
    compute_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_BCAST,
        CB_OUTPUT_TILES,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    f_addr, x_addr = input_tensor.buffer_address(), residual.buffer_address()
    p_addr, m_addr = post.buffer_address(), comb.buffer_address()
    o_addr = output_tensor.buffer_address()
    for core, start, count in assignment:
        reader_rt[core.x][core.y] = [f_addr, x_addr, p_addr, m_addr, start, count]
        writer_rt[core.x][core.y] = [o_addr, start, count]
        compute_rt[core.x][core.y] = [start, count]

    # UnpackToDestFp32 on every Float32 CB compute reads with copy_tile (derived from the CB format).
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    for index, dtype in (
        (CB_SUBLAYER_TILES, input_tensor.dtype),
        (CB_RESIDUAL_TILES, residual.dtype),
        (CB_COEF_BCAST, post.dtype),
    ):
        if dtype == ttnn.float32:
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
    )
    compute_cfg.dst_full_sync_en = compute_kernel_config.dst_full_sync_en
    compute_cfg.unpack_to_dest_mode = unpack_modes

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_cfg,
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
