# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_interleaved_to_sharded (owner-fetch regime).

One reader kernel on every core of the shard grid. Shard s goes to core G[s] of
``corerange_to_cores(grid, row_wise=ROW_MAJOR)``; that core reads its shard
rectangle from DRAM straight into its own L1 shard and drains with one barrier.
No CB, no compute kernel, no writer kernel.

``derive_plan`` is the single source of every symbol the kernel, the descriptor
and ``describe_plan`` use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

READER_KERNEL = Path(__file__).parent / "kernels" / "g6_interleaved_to_sharded_reader.cpp"

# Knobs. Each one is part of the descriptor cache key.
# Perf lamp L1: False moves every last-row page whole instead of its two trimmed face spans.
TRIM_LAST_ROW = True
# Perf lamp L3: the reader's RISC and NoC.
READER_PROCESSOR = ttnn.DataMovementProcessor.RISCV_1
READER_NOC = ttnn.NOC.NOC_0

# Runtime-arg layout shared with the reader kernel.
COMMON_ARG_INPUT_ADDR = 0
COMMON_ARG_OUTPUT_SHARD_BASE_ADDR = 1

_STANDARD_TILE_SHAPE = [32, 32]
_STANDARD_FACE_SHAPE = [16, 16]


@dataclass(frozen=True)
class CoreBlock:
    """The page rectangle one core of the shard grid receives (empty on an idle core)."""

    core: tuple[int, int]
    shard: int | None
    block_row_start: int
    block_rows: int
    block_col_start: int
    block_cols: int

    @property
    def role(self) -> str:
        return "idle" if self.shard is None else "owner"


@dataclass(frozen=True)
class Plan:
    tile_h: int
    tile_w: int
    tiles_h_per_image: int
    num_images: int
    tensor_rows_tiles: int
    tensor_cols_tiles: int
    last_row_valid_rows: int
    shard_rows_tiles: int
    shard_cols_tiles: int
    height_sharded: bool
    num_shards: int
    page_size: int
    face1_offset_bytes: int
    trim_span_bytes: int
    cores: tuple[CoreBlock, ...]

    @property
    def last_row_form(self) -> str:
        return "trimmed" if self.trim_span_bytes > 0 else "whole_page"


def tile_grid(input_tensor) -> tuple[int, int, int, int, int]:
    """(tile_h, tile_w, num_images, tiles_h_per_image, tensor_cols_tiles) of the input.

    Each image is tile-padded separately, so its tile rows are counted per image.
    """
    tile_h, tile_w = (int(d) for d in input_tensor.tile.tile_shape)
    padded = [int(d) for d in input_tensor.padded_shape]
    num_images = math.prod(padded[:-2])
    return tile_h, tile_w, num_images, padded[-2] // tile_h, padded[-1] // tile_w


def num_shards_of(
    height_sharded: bool, tensor_rows_tiles: int, tensor_cols_tiles: int, shard_rows_tiles: int, shard_cols_tiles: int
) -> int:
    if height_sharded:
        return math.ceil(tensor_rows_tiles / shard_rows_tiles)
    return math.ceil(tensor_cols_tiles / shard_cols_tiles)


def _trim_span_bytes(input_tensor, last_row_valid_rows: int) -> int:
    """Bytes moved per top face of a last-row page, or 0 when the page must move whole.

    Rows 0..15 of a standard 32x32 tile lie row-major in faces 0 and 1, so the first
    ``v`` rows are the first ``v * face_w`` elements of each top face. The span is
    rounded up to the DRAM read alignment, which the source and destination must share.
    """
    tile = input_tensor.tile
    face_h, face_w = (int(d) for d in tile.face_shape)
    trimmable = (
        TRIM_LAST_ROW
        and [int(d) for d in tile.tile_shape] == _STANDARD_TILE_SHAPE
        and [face_h, face_w] == _STANDARD_FACE_SHAPE
        and not tile.transpose_within_face
        and not tile.transpose_of_faces
        and last_row_valid_rows <= face_h
    )
    if not trimmable:
        return 0
    alignment = ttnn.get_dram_alignment()
    return (last_row_valid_rows * face_w * input_tensor.element_size() + alignment - 1) // alignment * alignment


def derive_plan(input_tensor, memory_config) -> Plan:
    tile_h, tile_w, num_images, tiles_h_per_image, tensor_cols_tiles = tile_grid(input_tensor)
    tensor_rows_tiles = num_images * tiles_h_per_image
    logical_h = int(input_tensor.shape[-2])
    last_row_valid_rows = logical_h - tile_h * (tiles_h_per_image - 1)

    shard_spec = memory_config.shard_spec
    shard_h, shard_w = (int(d) for d in shard_spec.shape)
    shard_rows_tiles, shard_cols_tiles = shard_h // tile_h, shard_w // tile_w
    height_sharded = memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    num_shards = num_shards_of(height_sharded, tensor_rows_tiles, tensor_cols_tiles, shard_rows_tiles, shard_cols_tiles)

    face_h, face_w = (int(d) for d in input_tensor.tile.face_shape)
    face1_offset_bytes = face_h * face_w * input_tensor.element_size()

    # The allocator's own core order for shard s -> G[s].
    shard_cores = ttnn.corerange_to_cores(
        shard_spec.grid, row_wise=shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    )
    cores = []
    for s, core in enumerate(shard_cores):
        xy = (core.x, core.y)
        if s >= num_shards:
            cores.append(CoreBlock(xy, None, 0, 0, 0, 0))
        elif height_sharded:
            row_start = s * shard_rows_tiles
            rows = min(shard_rows_tiles, tensor_rows_tiles - row_start)
            cores.append(CoreBlock(xy, s, row_start, rows, 0, tensor_cols_tiles))
        else:
            col_start = s * shard_cols_tiles
            cols = min(shard_cols_tiles, tensor_cols_tiles - col_start)
            cores.append(CoreBlock(xy, s, 0, tensor_rows_tiles, col_start, cols))

    return Plan(
        tile_h=tile_h,
        tile_w=tile_w,
        tiles_h_per_image=tiles_h_per_image,
        num_images=num_images,
        tensor_rows_tiles=tensor_rows_tiles,
        tensor_cols_tiles=tensor_cols_tiles,
        last_row_valid_rows=last_row_valid_rows,
        shard_rows_tiles=shard_rows_tiles,
        shard_cols_tiles=shard_cols_tiles,
        height_sharded=height_sharded,
        num_shards=num_shards,
        page_size=input_tensor.buffer_aligned_page_size(),
        face1_offset_bytes=face1_offset_bytes,
        trim_span_bytes=_trim_span_bytes(input_tensor, last_row_valid_rows),
        cores=tuple(cores),
    )


def describe(plan: Plan) -> dict:
    return {
        "regime": "owner_fetch",
        "last_row_form": plan.last_row_form,
        "trim_span_bytes": plan.trim_span_bytes,
        "num_shards": plan.num_shards,
        "cores": [
            {
                "core": block.core,
                "role": block.role,
                "shard": block.shard,
                "block_rows": block.block_rows,
                "block_cols": block.block_cols,
            }
            for block in plan.cores
        ],
    }


def _build(input_tensor, memory_config) -> ttnn.ProgramDescriptor:
    plan = derive_plan(input_tensor, memory_config)

    compile_time_args = [
        plan.tensor_cols_tiles,
        plan.tiles_h_per_image,
        plan.shard_cols_tiles,
        plan.page_size,
        plan.trim_span_bytes,
        plan.face1_offset_bytes,
    ]
    compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    runtime_args = ttnn.RuntimeArgs()
    for block in plan.cores:
        x, y = block.core
        runtime_args[x][y] = [block.block_row_start, block.block_rows, block.block_col_start, block.block_cols]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(READER_KERNEL),
        core_ranges=memory_config.shard_spec.grid,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        # Patched with the live addresses on every call.
        common_runtime_args=[0, 0],
        config=ttnn.DataMovementConfigDescriptor(processor=READER_PROCESSOR, noc=READER_NOC),
    )
    return ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[])


def _grid_key(grid) -> tuple:
    return tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in grid.ranges())


def cache_key(input_tensor, memory_config) -> tuple:
    """Everything that changes the program. The logical shape is in it because the
    logical H sets trim_span_bytes, which the padded shape does not show."""
    tile = input_tensor.tile
    input_memory = input_tensor.memory_config()
    shard_spec = memory_config.shard_spec
    return (
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        tuple(int(d) for d in tile.tile_shape),
        tuple(int(d) for d in tile.face_shape),
        bool(tile.transpose_within_face),
        bool(tile.transpose_of_faces),
        input_memory.buffer_type,
        input_memory.memory_layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        _grid_key(shard_spec.grid),
        tuple(int(d) for d in shard_spec.shape),
        shard_spec.orientation,
        ttnn.get_dram_alignment(),
        TRIM_LAST_ROW,
        READER_PROCESSOR,
        READER_NOC,
    )


_DESCRIPTOR_CACHE: dict[tuple, ttnn.ProgramDescriptor] = {}


def get_program_descriptor(input_tensor, output_tensor, memory_config) -> ttnn.ProgramDescriptor:
    """The cached descriptor for this call's key, with the two buffer addresses patched in."""
    output_shard_base_addr = output_tensor.buffer_address()
    dram_alignment = ttnn.get_dram_alignment()
    if output_shard_base_addr % dram_alignment != 0:
        # A DRAM read needs source and destination congruent modulo the DRAM alignment.
        raise RuntimeError(
            f"g6_interleaved_to_sharded: output shard base 0x{output_shard_base_addr:x} is not aligned to the "
            f"{dram_alignment} B DRAM read alignment"
        )

    key = cache_key(input_tensor, memory_config)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = _build(input_tensor, memory_config)
        _DESCRIPTOR_CACHE[key] = descriptor

    # nanobind returns copies of vector members, so the patched kernel is assigned back.
    kernels = descriptor.kernels
    common_args = [0, 0]
    common_args[COMMON_ARG_INPUT_ADDR] = input_tensor.buffer_address()
    common_args[COMMON_ARG_OUTPUT_SHARD_BASE_ADDR] = output_shard_base_addr
    kernels[0].common_runtime_args = common_args
    descriptor.kernels = kernels
    return descriptor
