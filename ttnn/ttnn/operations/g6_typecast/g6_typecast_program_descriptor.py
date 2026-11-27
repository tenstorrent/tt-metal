# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast program descriptor: regime selection, work split, CBs, kernels.

Built regime: tile_convert. Output page p is the conversion of input page p (same shape and layout,
so the page orders coincide); each core converts one contiguous range of the flattened page axis in
blocks of `block_tiles` pages through reader (NCRISC/NoC0) -> compute -> writer (BRISC/NoC1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Blocking knobs (op_design.md, Blocking Model). Every CB capacity, loop trip count and the block
# size the kernels see derive from these; none is restated elsewhere.
BLOCK_TILES_CAP = 8  # pages per block; cap below the whole per-core range so read/convert/write overlap
IN_DEPTH = 2  # cb_input_tiles depth in blocks
OUT_DEPTH = 2  # cb_output_tiles depth in blocks
PAGE_STRIDE = 1  # page walk stride; NB_L1 under the deferred bank-local regime
ROW_WISE = True  # core order of the split (row spreads beat column spreads on the NoC)

CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 1

CORE_ROLES = {
    "tile_convert": "reader+compute+writer",
    "tile_bitcopy": "reader+writer",
    "rm_convert": "reader+compute+writer",
    "rm_bitcopy": "reader+writer",
}

_BITCOPY_PAIRS = {(ttnn.uint32, ttnn.int32), (ttnn.int32, ttnn.uint32)}
# Typecast LLK endpoints that need 32-bit DEST.
_DEST32_INPUTS = {ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16}
_DEST32_OUTPUTS = {ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint8}
# SrcA keeps 19 bits; these inputs must bypass it to keep every bit.
_UNPACK_TO_DEST_INPUTS = {ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16}
_NUM_CIRCULAR_BUFFERS = 64  # covers WH (32) and BH (64)


def select_regime(layout, in_dtype, out_dtype) -> str:
    bitcopy = (in_dtype, out_dtype) in _BITCOPY_PAIRS
    if layout == ttnn.TILE_LAYOUT:
        return "tile_bitcopy" if bitcopy else "tile_convert"
    return "rm_bitcopy" if bitcopy else "rm_convert"


@dataclass(frozen=True)
class Plan:
    regime: str
    tensor_pages: int
    num_cores: int
    block_tiles: int
    all_cores: ttnn.CoreRangeSet
    # (core, first_page, core_tiles) in the order that assigns page ranges
    assignment: tuple


def make_plan(input_tensor: ttnn.Tensor, dtype: ttnn.DataType) -> Plan:
    regime = select_regime(input_tensor.layout, input_tensor.dtype, dtype)
    if regime != "tile_convert":
        raise NotImplementedError(f"g6_typecast: regime {regime} is not built")

    tensor_pages = input_tensor.buffer_num_pages()
    grid = input_tensor.device().compute_with_storage_grid_size()
    num_cores, all_cores, group_1, group_2, tiles_g1, tiles_g2 = ttnn.split_work_to_cores(
        grid, tensor_pages, row_wise=ROW_WISE
    )
    block_tiles = min(tiles_g1, BLOCK_TILES_CAP)

    groups = [(group_1, tiles_g1)]
    if tiles_g2 > 0:
        groups.append((group_2, tiles_g2))
    assignment = []
    first_page = 0
    for group, core_tiles in groups:
        for core in ttnn.corerange_to_cores(group, row_wise=ROW_WISE):
            assignment.append((core, first_page, core_tiles))
            first_page += core_tiles
    assert first_page == tensor_pages, f"g6_typecast: split covers {first_page} of {tensor_pages} pages"
    assert len(assignment) == num_cores

    return Plan(
        regime=regime,
        tensor_pages=tensor_pages,
        num_cores=num_cores,
        block_tiles=block_tiles,
        all_cores=all_cores,
        assignment=tuple(assignment),
    )


def _compute_config(in_dtype, out_dtype) -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor(
        math_approx_mode=False,
        fp32_dest_acc_en=(in_dtype in _DEST32_INPUTS or out_dtype in _DEST32_OUTPUTS),
        dst_full_sync_en=False,
        # The default bfp8 packer lands more than one quantization step off the host rendering on
        # some groups; the precise packer rounds as ttnn.from_torch does.
        bfp8_pack_precise=(out_dtype == ttnn.bfloat8_b),
    )
    if in_dtype in _UNPACK_TO_DEST_INPUTS:
        modes = [ttnn.UnpackToDestMode.Default] * _NUM_CIRCULAR_BUFFERS
        modes[CB_INPUT_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    return config


@dataclass
class CachedProgram:
    """Program parts reused across calls; only the two buffer addresses change between calls."""

    reader: ttnn.KernelDescriptor
    writer: ttnn.KernelDescriptor
    compute: ttnn.KernelDescriptor
    cbs: list

    def descriptor(self, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.ProgramDescriptor:
        self.reader.common_runtime_args = [input_tensor.buffer_address()]
        self.writer.common_runtime_args = [output_tensor.buffer_address()]
        return ttnn.ProgramDescriptor(kernels=[self.reader, self.writer, self.compute], semaphores=[], cbs=self.cbs)


def build_program(plan: Plan, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> CachedProgram:
    assert output_tensor.buffer_num_pages() == plan.tensor_pages, "input and output page orders must coincide"

    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    block_tiles = plan.block_tiles
    all_cores = plan.all_cores

    cb_input_tiles = ttnn.CBDescriptor(
        total_size=IN_DEPTH * block_tiles * in_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page_bytes
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=OUT_DEPTH * block_tiles * out_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES, data_format=output_tensor.dtype, page_size=out_page_bytes
            )
        ],
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core, first_page, core_tiles in plan.assignment:
        reader_rt[core.x][core.y] = [first_page, core_tiles, PAGE_STRIDE]
        writer_rt[core.x][core.y] = [first_page, core_tiles, PAGE_STRIDE]
        compute_rt[core.x][core.y] = [core_tiles]

    reader_ct = [CB_INPUT_TILES, block_tiles, in_page_bytes]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [CB_OUTPUT_TILES, block_tiles, out_page_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct = [CB_INPUT_TILES, CB_OUTPUT_TILES, block_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=_compute_config(input_tensor.dtype, output_tensor.dtype),
    )
    return CachedProgram(reader=reader, writer=writer, compute=compute, cbs=[cb_input_tiles, cb_output_tiles])
