# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for clamp_square_practice (regime `interleaved_stream`).

Blocking knobs (op_design.md -> Blocking Model), each defined once here:
  BLOCK_TILES_DEFAULT  - tiles per block (one CB handshake / NoC barrier / DEST window);
                         clamped by the DEST capacity and by the per-core assignment.
  INPUT_DEPTH_BLOCKS   - cb_input_tiles capacity in blocks.
  OUTPUT_DEPTH_BLOCKS  - cb_output_tiles capacity in blocks.
                         Perf 1: 32 blocks deep (was 2) so compute hides under the DRAM-bound DM;
                         each CB is capped at the per-core assignment rounded up to a whole block
                         (a CB that holds the whole assignment never blocks its producer), so
                         small shapes pay no extra L1. Max 2 x 256 bf16 tiles = 1 MiB per core.
  core_grid            - builder parameter; defaults to the full compute grid.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

BLOCK_TILES_DEFAULT = 8
INPUT_DEPTH_BLOCKS = 32
OUTPUT_DEPTH_BLOCKS = 32

FP32_DEST_ACC_EN = False
DST_FULL_SYNC_EN = False

CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 1


def dest_limit(fp32_dest_acc_en: bool, dst_full_sync_en: bool) -> int:
    """Mirror of get_dest_limit() in kernel_lib/dest_helpers.hpp (16-bit DEST, half-sync = 8)."""
    tiles = 16 if dst_full_sync_en else 8
    return tiles // 2 if fp32_dest_acc_en else tiles


def _fp32_bits(value: float) -> int:
    """IEEE-754 fp32 bit pattern. validate() guarantees `value` is within fp32 range."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    min_val: float,
    max_val: float,
    *,
    core_grid=None,
    block_tiles_default: int = BLOCK_TILES_DEFAULT,
    input_depth_blocks: int = INPUT_DEPTH_BLOCKS,
    output_depth_blocks: int = OUTPUT_DEPTH_BLOCKS,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    if core_grid is None:
        core_grid = device.compute_with_storage_grid_size()

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()
    tensor_tiles = input_tensor.buffer_num_pages()
    assert output_tensor.buffer_num_pages() == tensor_tiles

    # ---- Work distribution: contiguous page ranges along the flattened tile axis ----
    (num_cores, all_cores, core_group_1, core_group_2, core_tiles_g1, core_tiles_g2) = ttnn.split_work_to_cores(
        core_grid, tensor_tiles, row_wise=True
    )

    # ---- Block extent (single source of truth) ----
    block_tiles = max(1, min(block_tiles_default, dest_limit(FP32_DEST_ACC_EN, DST_FULL_SYNC_EN), core_tiles_g1))

    # ---- Circular buffers ----
    # Depth in tiles, capped at the largest per-core assignment rounded up to a whole block.
    assignment_tiles = -(-core_tiles_g1 // block_tiles) * block_tiles
    input_cb_tiles = min(input_depth_blocks * block_tiles, assignment_tiles)
    output_cb_tiles = min(output_depth_blocks * block_tiles, assignment_tiles)
    cb_input_tiles = ttnn.CBDescriptor(
        total_size=input_cb_tiles * input_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=input_page_size
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=output_cb_tiles * output_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES, data_format=output_tensor.dtype, page_size=output_page_size
            )
        ],
    )

    # ---- Per-core runtime args ----
    min_bits = _fp32_bits(min_val)
    max_bits = _fp32_bits(max_val)
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    g1_cores = set((c.x, c.y) for c in ttnn.corerange_to_cores(core_group_1, row_wise=True))
    start_tile = 0
    for core in ttnn.corerange_to_cores(all_cores, row_wise=True):
        core_tiles = core_tiles_g1 if (core.x, core.y) in g1_cores else core_tiles_g2
        reader_rt_args[core.x][core.y] = [in_addr, start_tile, core_tiles]
        writer_rt_args[core.x][core.y] = [out_addr, start_tile, core_tiles]
        compute_rt_args[core.x][core.y] = [core_tiles, min_bits, max_bits]
        start_tile += core_tiles
    assert start_tile == tensor_tiles

    # ---- Kernels ----
    reader_ct_args = [CB_INPUT_TILES, block_tiles, input_page_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [CB_OUTPUT_TILES, block_tiles, output_page_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct_args = [CB_INPUT_TILES, CB_OUTPUT_TILES, block_tiles]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "clamp_square_practice_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "clamp_square_practice_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "clamp_square_practice_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=FP32_DEST_ACC_EN, dst_full_sync_en=DST_FULL_SYNC_EN),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_input_tiles, cb_output_tiles],
    )
