# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast — ProgramDescriptor for the TILE stream regime (op_design.md, Blocking Model).

Pipeline per core (contiguous page range from split_work_to_cores over the full grid):
    reader  (NCRISC, NoC0): input pages  -> cb_input_tiles   one barrier per block
    compute (TRISC)       : cb_input_tiles -> typecast chain -> cb_output_tiles
                            (one helper call over the whole range, CB-synced per block)
    writer  (BRISC, NoC1) : cb_output_tiles -> output pages  one flush per block, one final barrier

Knobs (single source each; every dependent quantity derives from them):
    BLOCK_TILES_DEFAULT  pages per block (CB handshake / NoC barrier granularity)
    IN_DEPTH_BLOCKS      cb_input_tiles depth in blocks
    OUT_DEPTH_BLOCKS     cb_output_tiles depth in blocks
    SPLIT_ROW_WISE       core-assignment order of split_work_to_cores

BB-07: the descriptor is built once per call signature and cached; a repeat call only refreshes
the two buffer addresses (common runtime args of reader / writer).
BB-08: last_plan() reports the plan of the most recent call.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Blocking-model knobs (op_design.md) ----
BLOCK_TILES_DEFAULT = 8
IN_DEPTH_BLOCKS = 2
OUT_DEPTH_BLOCKS = 2
SPLIT_ROW_WISE = True

# ---- CB slots (semantic names) ----
CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 16

# Dtypes whose values need a 32-bit DEST (typecast.h: "For input/output to be UInt32, Int32, or
# Float32, Dest must be in 32 bit mode").
_DTYPES_32BIT = (ttnn.float32, ttnn.int32, ttnn.uint32)

# Kernel indices inside the ProgramDescriptor (address refresh on a cache hit).
_READER_IDX, _WRITER_IDX, _COMPUTE_IDX = 0, 1, 2

_PROGRAM_CACHE = {}
_LAST_PLAN = None


def last_plan():
    """The plan the most recent g6_typecast call selected (BB-08)."""
    return _LAST_PLAN


def _cache_key(input_tensor, output_tensor, grid):
    in_mc = input_tensor.memory_config()
    out_mc = output_tensor.memory_config()
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        output_tensor.dtype,
        input_tensor.layout,
        in_mc.buffer_type,
        in_mc.memory_layout,
        out_mc.buffer_type,
        out_mc.memory_layout,
        input_tensor.device().id(),
        (grid.x, grid.y),
        BLOCK_TILES_DEFAULT,
        IN_DEPTH_BLOCKS,
        OUT_DEPTH_BLOCKS,
        SPLIT_ROW_WISE,
    )


def get_program(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.ProgramDescriptor:
    """Cached descriptor for this call signature, with the buffer addresses of this call."""
    global _LAST_PLAN
    grid = input_tensor.device().compute_with_storage_grid_size()
    key = _cache_key(input_tensor, output_tensor, grid)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        entry = create_program_descriptor(input_tensor, output_tensor, grid)
        _PROGRAM_CACHE[key] = entry
    descriptor, plan = entry
    descriptor.kernels[_READER_IDX].common_runtime_args = [input_tensor.buffer_address()]
    descriptor.kernels[_WRITER_IDX].common_runtime_args = [output_tensor.buffer_address()]
    _LAST_PLAN = plan
    return descriptor


def create_program_descriptor(input_tensor, output_tensor, grid):
    """Build (descriptor, plan) for the TILE stream regime."""
    # ---- Page axis (leading dims, tile rows, tile cols folded) ----
    tensor_pages = output_tensor.buffer_num_pages()
    assert tensor_pages == input_tensor.buffer_num_pages(), "g6_typecast: input/output page counts differ"
    in_page_bytes = input_tensor.buffer_aligned_page_size()
    out_page_bytes = output_tensor.buffer_aligned_page_size()

    # ---- Core assignment: contiguous page ranges over the full grid ----
    (num_cores, all_cores, core_group_1, core_group_2, pages_per_core_1, pages_per_core_2) = ttnn.split_work_to_cores(
        grid, tensor_pages, row_wise=SPLIT_ROW_WISE
    )
    core_pages_max = max(pages_per_core_1, pages_per_core_2)

    # ---- Block extent (one knob; CB capacities derive from it) ----
    block_tiles = min(core_pages_max, BLOCK_TILES_DEFAULT)
    in_cb_pages = IN_DEPTH_BLOCKS * block_tiles
    out_cb_pages = OUT_DEPTH_BLOCKS * block_tiles

    # ---- Circular buffers ----
    cb_input_tiles = ttnn.CBDescriptor(
        total_size=in_cb_pages * in_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page_bytes
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=out_cb_pages * out_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES, data_format=output_tensor.dtype, page_size=out_page_bytes
            )
        ],
    )

    # ---- Per-core runtime args: (start_page, page_stride, core_pages) ----
    page_stride = 1  # contiguous ranges; the bank-affine regime turns this knob
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start_page = 0
    for group, core_pages in ((core_group_1, pages_per_core_1), (core_group_2, pages_per_core_2)):
        if core_pages == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, SPLIT_ROW_WISE):
            reader_rt[core.x][core.y] = [start_page, page_stride, core_pages]
            writer_rt[core.x][core.y] = [start_page, page_stride, core_pages]
            compute_rt[core.x][core.y] = [core_pages]
            start_page += core_pages
    assert start_page == tensor_pages, "g6_typecast: work split does not cover the tensor"

    # ---- Kernels ----
    reader_ct = [CB_INPUT_TILES, block_tiles, in_page_bytes]
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

    fp32_dest_acc_en = input_tensor.dtype in _DTYPES_32BIT or output_tensor.dtype in _DTYPES_32BIT
    # Block-float destination: pack from a Float16_b source so the packer rounds each group to the
    # shared exponent (the default Bfp8_b pack source misses the one-step gate, design risk R5).
    bfp8_pack_precise = output_tensor.dtype == ttnn.bfloat8_b
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[CB_INPUT_TILES, CB_OUTPUT_TILES, block_tiles],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en, bfp8_pack_precise=bfp8_pack_precise),
    )

    kernels = [None, None, None]
    kernels[_READER_IDX], kernels[_WRITER_IDX], kernels[_COMPUTE_IDX] = reader, writer, compute
    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_input_tiles, cb_output_tiles])

    plan = {
        "regime": "tile_stream",
        "num_cores": num_cores,
        "tensor_pages": tensor_pages,
        "pages_per_core": (pages_per_core_1, pages_per_core_2),
        "block_tiles": block_tiles,
        "in_depth_blocks": IN_DEPTH_BLOCKS,
        "out_depth_blocks": OUT_DEPTH_BLOCKS,
        "fp32_dest_acc_en": fp32_dest_acc_en,
        "bfp8_pack_precise": bfp8_pack_precise,
        "core_roles": "reader+compute+writer on every core",
        "l1_bytes_per_core": in_cb_pages * in_page_bytes + out_cb_pages * out_page_bytes,
    }
    return descriptor, plan
