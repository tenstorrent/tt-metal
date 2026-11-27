# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast plan (blocking model) and the ProgramDescriptor built from it.

`_plan()` is the single source of every block extent, buffer depth and core assignment;
the descriptor, the CB sizes and every kernel argument are derived from the Plan it returns.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 1

# L1 the model leaves to the op on every core while it runs (BB-05).
L1_OP_BUDGET = 1_347_776

REINTERPRET_PAIRS = {(ttnn.uint32, ttnn.int32), (ttnn.int32, ttnn.uint32)}
BUILT_REGIMES = ("tile_convert",)

# Int32/Float32 pack (and an INT32 SFPU store) needs 32-bit DEST; from 16-bit DEST it corrupts silently.
_DTYPES_NEEDING_FP32_DEST = (ttnn.float32, ttnn.int32, ttnn.uint32)


@dataclasses.dataclass(frozen=True)
class Knobs:
    """Tunables of the blocking scheme. Defaults are the design's Phase 0 values; every field
    enters the program cache key."""

    max_cores: int | None = None  # None: the whole compute grid
    pipeline_depth: int = 2
    block_tiles: int | None = None  # None: the Phase 0 extent rule in _plan()
    page_stride: int = 1


@dataclasses.dataclass(frozen=True)
class CoreWork:
    x: int
    y: int
    first_page: int
    core_tiles: int
    num_blocks: int
    last_block_extent: int


@dataclasses.dataclass(frozen=True)
class Plan:
    regime: str
    tensor_tiles: int
    max_cores: int
    num_cores: int
    all_cores: ttnn.CoreRangeSet
    core_tiles_max: int
    pipeline_depth: int
    block_tiles: int
    block_tiles_l1_cap: int
    page_stride: int
    in_page_bytes: int
    out_page_bytes: int
    out_bank_bytes: int
    cb_input_bytes: int
    cb_output_bytes: int
    fp32_dest_acc_en: bool
    bfp8_pack_precise: bool
    cores: tuple[CoreWork, ...]


def select_regime(layout, dtype, output_dtype, buffer, in_aligned_page_bytes, dram_alignment):
    if (dtype, output_dtype) in REINTERPRET_PAIRS:
        return "tile_reinterpret" if layout == ttnn.TILE_LAYOUT else "rm_reinterpret"
    if layout == ttnn.TILE_LAYOUT:
        return "tile_convert"
    if buffer == "dram" and in_aligned_page_bytes == dram_alignment:
        return "rm_convert_bank_runs"
    return "rm_convert"


def _buffer_name(tensor):
    return "l1" if tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def _num_l1_banks(device, grid):
    try:
        return ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.L1).num_banks
    except Exception:
        # Fewer banks overstates each bank's share of the output: the L1 cap stays conservative.
        return grid.x * grid.y


def _ceil_div(a, b):
    return -(-a // b)


def _plan(input_tensor, output_dtype, knobs: Knobs = Knobs()) -> Plan:
    device = input_tensor.device()
    buffer = _buffer_name(input_tensor)
    regime = select_regime(
        input_tensor.layout,
        input_tensor.dtype,
        output_dtype,
        buffer,
        input_tensor.buffer_aligned_page_size(),
        ttnn.get_dram_alignment(),
    )
    if regime not in BUILT_REGIMES:
        raise NotImplementedError(f"g6_typecast: regime {regime!r} is not built")
    if knobs.page_stride != 1:
        # A strided walk needs the bank-affine core assignment (regime l1_bank_affine), not built yet.
        raise ValueError(f"g6_typecast: page_stride={knobs.page_stride} needs a bank-affine assignment")
    if knobs.pipeline_depth < 1:
        raise ValueError(f"g6_typecast: pipeline_depth must be >= 1, got {knobs.pipeline_depth}")

    tensor_tiles = input_tensor.buffer_num_pages()
    in_page_bytes = input_tensor.buffer_page_size()
    # The output keeps the input's layout and tile, so its page is one tile in the output dtype.
    out_page_bytes = input_tensor.tile.get_tile_size(output_dtype)

    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    max_cores = grid_cores if knobs.max_cores is None else max(1, min(knobs.max_cores, grid_cores))
    core_pool = ttnn.num_cores_to_corerangeset(max_cores, grid, True)
    num_cores, all_cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(core_pool, tensor_tiles, True)
    core_tiles_max = _ceil_div(tensor_tiles, num_cores)

    pipeline_depth = knobs.pipeline_depth
    out_bank_bytes = 0
    if buffer == "l1":
        out_bank_bytes = _ceil_div(tensor_tiles, _num_l1_banks(device, grid)) * out_page_bytes
    block_tiles_l1_cap = (L1_OP_BUDGET - out_bank_bytes) // (pipeline_depth * (in_page_bytes + out_page_bytes))
    if block_tiles_l1_cap < 1:
        raise RuntimeError(
            f"g6_typecast: L1 budget {L1_OP_BUDGET} B/core cannot hold one block "
            f"(output bank share {out_bank_bytes} B, pipeline_depth {pipeline_depth})"
        )
    if knobs.block_tiles is None:
        block_tiles = min(_ceil_div(core_tiles_max, pipeline_depth), block_tiles_l1_cap)
    else:
        block_tiles = knobs.block_tiles
    block_tiles = max(1, min(block_tiles, core_tiles_max, block_tiles_l1_cap))

    cores = []
    first_page = 0
    for group, group_tiles in ((group_1, tiles_1), (group_2, tiles_2)):
        if group_tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            num_blocks = _ceil_div(group_tiles, block_tiles)
            cores.append(
                CoreWork(
                    x=core.x,
                    y=core.y,
                    first_page=first_page,
                    core_tiles=group_tiles,
                    num_blocks=num_blocks,
                    last_block_extent=group_tiles - (num_blocks - 1) * block_tiles,
                )
            )
            first_page += group_tiles
    assert first_page == tensor_tiles and len(cores) == num_cores

    fp32_dest_acc_en = input_tensor.dtype in _DTYPES_NEEDING_FP32_DEST or output_dtype in _DTYPES_NEEDING_FP32_DEST

    return Plan(
        regime=regime,
        tensor_tiles=tensor_tiles,
        max_cores=max_cores,
        num_cores=num_cores,
        all_cores=all_cores,
        core_tiles_max=core_tiles_max,
        pipeline_depth=pipeline_depth,
        block_tiles=block_tiles,
        block_tiles_l1_cap=block_tiles_l1_cap,
        page_stride=knobs.page_stride,
        in_page_bytes=in_page_bytes,
        out_page_bytes=out_page_bytes,
        out_bank_bytes=out_bank_bytes,
        cb_input_bytes=pipeline_depth * block_tiles * in_page_bytes,
        cb_output_bytes=pipeline_depth * block_tiles * out_page_bytes,
        fp32_dest_acc_en=fp32_dest_acc_en,
        bfp8_pack_precise=output_dtype == ttnn.bfloat8_b,
        cores=tuple(cores),
    )


READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1
COMPUTE_KERNEL_INDEX = 2


# The buffer address is the only per-call value of the program; it lives in the dataflow kernels'
# common runtime args so a program-cache hit rewrites two small vectors (patch_addresses).
def _reader_common_args(input_tensor, plan: Plan):
    return [input_tensor.buffer_address(), plan.block_tiles, plan.page_stride]


def _writer_common_args(output_tensor, plan: Plan):
    return [output_tensor.buffer_address(), plan.block_tiles, plan.page_stride]


def patch_addresses(descriptor, plan: Plan, input_tensor, output_tensor):
    descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args = _reader_common_args(input_tensor, plan)
    descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args = _writer_common_args(output_tensor, plan)


def create_program_descriptor(input_tensor, output_tensor, plan: Plan) -> ttnn.ProgramDescriptor:
    assert output_tensor.buffer_num_pages() == plan.tensor_tiles
    assert output_tensor.buffer_page_size() == plan.out_page_bytes

    cb_input_tiles = ttnn.CBDescriptor(
        total_size=plan.cb_input_bytes,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES,
                data_format=input_tensor.dtype,
                page_size=plan.in_page_bytes,
            )
        ],
    )
    cb_output_tiles = ttnn.CBDescriptor(
        total_size=plan.cb_output_bytes,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=plan.out_page_bytes,
            )
        ],
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for w in plan.cores:
        reader_rt[w.x][w.y] = [w.first_page, w.num_blocks, w.last_block_extent]
        writer_rt[w.x][w.y] = [w.first_page, w.num_blocks, w.last_block_extent]
        compute_rt[w.x][w.y] = [w.num_blocks, w.last_block_extent]

    reader_ct = [CB_INPUT_TILES, plan.in_page_bytes]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=_reader_common_args(input_tensor, plan),
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct = [CB_OUTPUT_TILES, plan.out_page_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=_writer_common_args(output_tensor, plan),
        config=ttnn.WriterConfigDescriptor(),
    )

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_typecast_compute.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=[CB_INPUT_TILES, CB_OUTPUT_TILES],
        runtime_args=compute_rt,
        common_runtime_args=[plan.block_tiles],
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=plan.fp32_dest_acc_en, bfp8_pack_precise=plan.bfp8_pack_precise
        ),
    )

    kernels = [None, None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    kernels[COMPUTE_KERNEL_INDEX] = compute
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_input_tiles, cb_output_tiles])
