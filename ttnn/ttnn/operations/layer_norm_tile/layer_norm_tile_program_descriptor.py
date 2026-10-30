# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm ProgramDescriptor: regime row-split-resident-w (tile-rows split across the grid, whole W resident)."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FACE = 16

# Blocking-model host knobs (op_design.md, Blocking Model).
BLOCK_ROWS_MAX = 2
INPUT_DEPTH_BLOCKS_MAX = 2
OUTPUT_DEPTH_ROWS = 2
W_TILES_RESIDENT_MAX = 32

CB_INPUT_TILES = 0
CB_GAMMA_ROWS = 1
CB_BETA_ROWS = 2
CB_ROW_STAT = 3
CB_RSTD = 4
CB_CENTERED = 5
CB_OUTPUT_TILES = 6


def _div_up(a, b):
    return -(-a // b)


def affine_half_row_bytes(affine):
    """Bytes of one face-row (16 elements): the unit a tile's row 0 is split into."""
    return FACE * affine.element_size()


@dataclass(frozen=True)
class Blocking:
    block_rows: int
    block_w_tiles: int
    input_depth_blocks: int
    output_depth_rows: int
    all_cores: object
    assignment: tuple  # ((core, start_row_tile, core_rows, num_blocks), ...)


def _tensor_row_tiles(shape):
    leading = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return leading * _div_up(shape[-2], TILE)


def select_blocking(input_tensor):
    """Single source of every block knob: block_rows, block_w_tiles, buffer depths, core assignment."""
    shape = list(input_tensor.shape)
    tensor_row_tiles = _tensor_row_tiles(shape)
    block_w_tiles = _div_up(shape[-1], TILE)

    grid = input_tensor.device().compute_with_storage_grid_size()
    (_, all_cores, group_1, group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, tensor_row_tiles, row_wise=True)

    block_rows = min(rows_g1, BLOCK_ROWS_MAX)
    max_num_blocks = _div_up(rows_g1, block_rows)
    input_depth_blocks = min(INPUT_DEPTH_BLOCKS_MAX, max_num_blocks)

    assignment = []
    start = 0
    for group, core_rows in ((group_1, rows_g1), (group_2, rows_g2)):
        if core_rows == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, core_rows, _div_up(core_rows, block_rows)))
            start += core_rows
    assert start == tensor_row_tiles

    return Blocking(
        block_rows=block_rows,
        block_w_tiles=block_w_tiles,
        input_depth_blocks=input_depth_blocks,
        output_depth_rows=OUTPUT_DEPTH_ROWS,
        all_cores=all_cores,
        assignment=tuple(assignment),
    )


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cb(index, num_pages, page_size, data_format, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _affine_reader_args(affine):
    """(is_tile, stick_elems, read_bytes, page_bytes) for one affine tensor's row-0 reads."""
    half_row = affine_half_row_bytes(affine)
    if affine.layout == ttnn.TILE_LAYOUT:
        alignment = (
            ttnn.get_dram_alignment()
            if affine.memory_config().buffer_type == ttnn.BufferType.DRAM
            else ttnn.get_l1_alignment()
        )
        # One read per face row; a wider NoC granule drags in the following face rows, which are ignored.
        read_bytes = max(half_row, alignment)
        assert read_bytes <= FACE * FACE * affine.element_size()
        return [1, 0, read_bytes, affine.buffer_aligned_page_size()]
    return [0, list(affine.shape)[-1], half_row, affine.buffer_aligned_page_size()]


def create_program_descriptor(input_tensor, gamma, beta, output_tensor, blocking, *, epsilon, compute_kernel_config):
    width = list(input_tensor.shape)[-1]
    block_rows = blocking.block_rows
    block_w_tiles = blocking.block_w_tiles
    block_tiles = block_rows * block_w_tiles
    cores = blocking.all_cores

    input_tile_bytes = input_tensor.tile.get_tile_size(input_tensor.dtype)
    output_tile_bytes = output_tensor.tile.get_tile_size(output_tensor.dtype)
    affine_tile_bytes = ttnn.tile_size(gamma.dtype)
    assert ttnn.tile_size(beta.dtype) == affine_tile_bytes
    intermediate_format = ttnn.float32 if compute_kernel_config.fp32_dest_acc_en else ttnn.bfloat16
    intermediate_tile_bytes = ttnn.tile_size(intermediate_format)

    cbs = [
        _cb(CB_INPUT_TILES, blocking.input_depth_blocks * block_tiles, input_tile_bytes, input_tensor.dtype, cores),
        _cb(CB_GAMMA_ROWS, block_w_tiles, affine_tile_bytes, gamma.dtype, cores),
        _cb(CB_BETA_ROWS, block_w_tiles, affine_tile_bytes, beta.dtype, cores),
        _cb(CB_ROW_STAT, block_rows, intermediate_tile_bytes, intermediate_format, cores),
        _cb(CB_RSTD, block_rows, intermediate_tile_bytes, intermediate_format, cores),
        _cb(CB_CENTERED, block_tiles, intermediate_tile_bytes, intermediate_format, cores),
        _cb(
            CB_OUTPUT_TILES,
            blocking.output_depth_rows * block_w_tiles,
            output_tile_bytes,
            output_tensor.dtype,
            cores,
        ),
    ]

    reader_ct_args = [
        CB_INPUT_TILES,
        CB_GAMMA_ROWS,
        CB_BETA_ROWS,
        block_rows,
        block_w_tiles,
        input_tile_bytes,
        affine_tile_bytes // 4,  # face bytes: row 0 of face 1 sits one face into the tile
        affine_half_row_bytes(gamma),
    ]
    reader_ct_args.extend(_affine_reader_args(gamma))
    reader_ct_args.extend(_affine_reader_args(beta))
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    writer_ct_args = [CB_OUTPUT_TILES, block_rows, block_w_tiles, output_tile_bytes]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [
        CB_INPUT_TILES,
        CB_GAMMA_ROWS,
        CB_BETA_ROWS,
        CB_ROW_STAT,
        CB_RSTD,
        CB_CENTERED,
        CB_OUTPUT_TILES,
        block_rows,
        block_w_tiles,
        width,
        _float_bits(1.0 / width),
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address()
    beta_addr = beta.buffer_address()
    eps_bits = _float_bits(epsilon)
    for core, start_row_tile, core_rows, num_blocks in blocking.assignment:
        reader_rt[core.x][core.y] = [in_addr, gamma_addr, beta_addr, start_row_tile, core_rows, num_blocks]
        writer_rt[core.x][core.y] = [out_addr, start_row_tile, core_rows, num_blocks]
        compute_rt[core.x][core.y] = [num_blocks, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
