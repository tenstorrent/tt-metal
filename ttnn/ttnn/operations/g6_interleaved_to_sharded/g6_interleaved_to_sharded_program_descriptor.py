# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_interleaved_to_sharded.

Owner-fetch: shard s goes to core G[s] (the allocator's own core order), and that core reads its
clipped shard rectangle of DRAM-interleaved pages straight into its L1 shard. One reader kernel on
NCRISC/NoC0, no CB, no writer, no compute kernel. `plan()` is the single source of every knob; the
builder and `describe_plan()` both read it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
READER_KERNEL = KERNEL_DIR / "g6_interleaved_to_sharded_reader.cpp"

# How a partial tile row (rows >= H % tile_h of an image's last tile row) crosses DRAM:
#   "valid_rows": only the face rows that hold valid data, rounded up to the DRAM read granule.
#   "whole_page": the whole page, padding included.
PARTIAL_TILE_TRANSFER = "valid_rows"
_PARTIAL_TILE_TRANSFERS = ("valid_rows", "whole_page")

# Free L1 per core the model leaves to the op (BB-05); the output shard counts inside it.
L1_BUDGET_BYTES = 1347776

_MAX_PARTIAL_SEGMENTS = 2


@dataclass(frozen=True)
class CoreBlock:
    core: tuple  # (x, y)
    shard: int
    r0: int
    c0: int
    block_rows: int
    block_cols: int


@dataclass(frozen=True)
class Plan:
    scheme: str  # "height_sharded" | "width_sharded"
    page_bytes: int
    tile_h: int
    tensor_rows_per_image_tiles: int
    tensor_rows_tiles: int
    tensor_cols_tiles: int
    last_row_valid_rows: int
    shard_rows_tiles: int
    shard_cols_tiles: int
    num_shards: int
    partial_tile_transfer: str
    # (offset, bytes) pairs read for a partial tile row; (0, page_bytes) for a whole-page transfer.
    partial_segments: tuple
    blocks: tuple

    @property
    def last_row_is_partial(self) -> bool:
        return self.last_row_valid_rows < self.tile_h


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _face_geometry_supported(tile, dtype) -> bool:
    return (
        list(tile.tile_shape) == [32, 32]
        and list(tile.face_shape) == [16, 16]
        and not tile.transpose_of_faces
        and not tile.transpose_within_face
        and dtype == ttnn.bfloat16
    )


def _valid_rows_segments(valid_rows: int, tile, datum_bytes: int, granule: int) -> tuple:
    """Byte ranges of a tile page that hold its first `valid_rows` rows.

    Faces are stored f0 (top-left), f1 (top-right), f2 (bottom-left), f3 (bottom-right), each
    row-major. A tile row lies in two faces, so each face contributes its own leading rows. Each
    face's range is rounded up to the DRAM read granule (Blackhole needs dst == src mod 64, and both
    face offsets and shard slots are 64-aligned); contiguous ranges are merged.
    """
    tile_h, tile_w = tile.tile_shape
    face_h, face_w = tile.face_shape
    face_row_bytes = face_w * datum_bytes
    face_bytes = face_h * face_row_bytes
    faces_per_row = tile_w // face_w
    segments = []
    for face_row in range(tile_h // face_h):
        rows_in_face = min(max(valid_rows - face_row * face_h, 0), face_h)
        if rows_in_face == 0:
            continue
        size = min(_align_up(rows_in_face * face_row_bytes, granule), face_bytes)
        for face_col in range(faces_per_row):
            offset = (face_row * faces_per_row + face_col) * face_bytes
            if segments and segments[-1][0] + segments[-1][1] == offset:
                segments[-1] = (segments[-1][0], segments[-1][1] + size)
            else:
                segments.append((offset, size))
    return tuple(segments)


def _active_cores(shard_spec, num_shards: int) -> list:
    row_wise = shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    grid = shard_spec.grid
    cores = ttnn.corerange_to_cores(grid, grid.num_cores(), row_wise)
    return [(c.x, c.y) for c in cores[:num_shards]]


def tile_grid(input_tensor):
    """(images, tile rows per image, tile cols, valid rows of an image's last tile row)."""
    shape = list(input_tensor.shape)
    tile_h, tile_w = input_tensor.tile.tile_shape
    height, width = shape[-2], shape[-1]
    images = math.prod(shape[:-2]) if len(shape) > 2 else 1
    rows_per_image = -(-height // tile_h)
    cols = -(-width // tile_w)
    last_row_valid_rows = height - tile_h * (rows_per_image - 1)
    return images, rows_per_image, cols, last_row_valid_rows


def plan(input_tensor, memory_config, partial_tile_transfer: str = PARTIAL_TILE_TRANSFER) -> Plan:
    if partial_tile_transfer not in _PARTIAL_TILE_TRANSFERS:
        raise ValueError(f"partial_tile_transfer must be one of {_PARTIAL_TILE_TRANSFERS}")
    tile = input_tensor.tile
    tile_h, tile_w = tile.tile_shape
    page_bytes = input_tensor.buffer_page_size()
    images, rows_per_image, cols, last_row_valid_rows = tile_grid(input_tensor)
    rows = images * rows_per_image

    shard_spec = memory_config.shard_spec
    shard_rows_tiles = shard_spec.shape[0] // tile_h
    shard_cols_tiles = shard_spec.shape[1] // tile_w
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        scheme = "height_sharded"
        num_shards = -(-rows // shard_rows_tiles)
    else:
        scheme = "width_sharded"
        num_shards = -(-cols // shard_cols_tiles)

    if (
        partial_tile_transfer == "valid_rows"
        and last_row_valid_rows < tile_h
        and _face_geometry_supported(tile, input_tensor.dtype)
    ):
        segments = _valid_rows_segments(
            last_row_valid_rows, tile, input_tensor.element_size(), ttnn.get_dram_alignment()
        )
        transfer = "valid_rows"
    else:
        segments = ((0, page_bytes),)
        transfer = "whole_page"
    assert 1 <= len(segments) <= _MAX_PARTIAL_SEGMENTS, segments

    blocks = []
    for s, core in enumerate(_active_cores(shard_spec, num_shards)):
        if scheme == "height_sharded":
            r0, c0 = s * shard_rows_tiles, 0
        else:
            r0, c0 = 0, s * shard_cols_tiles
        blocks.append(
            CoreBlock(
                core=core,
                shard=s,
                r0=r0,
                c0=c0,
                block_rows=min(shard_rows_tiles, rows - r0),
                block_cols=min(shard_cols_tiles, cols - c0),
            )
        )

    return Plan(
        scheme=scheme,
        page_bytes=page_bytes,
        tile_h=tile_h,
        tensor_rows_per_image_tiles=rows_per_image,
        tensor_rows_tiles=rows,
        tensor_cols_tiles=cols,
        last_row_valid_rows=last_row_valid_rows,
        shard_rows_tiles=shard_rows_tiles,
        shard_cols_tiles=shard_cols_tiles,
        num_shards=num_shards,
        partial_tile_transfer=transfer,
        partial_segments=segments,
        blocks=tuple(blocks),
    )


def _core_ranges(cores: list) -> ttnn.CoreRangeSet:
    """The active cores as few rectangles (BB-17): the first num_shards cores of the shard grid in
    orientation order are whole grid rows (or columns) plus at most one partial one."""
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores]
    return ttnn.CoreRangeSet(ranges[:1]).merge(ttnn.CoreRangeSet(ranges))


def describe_plan(input_tensor, memory_config) -> dict:
    p = plan(input_tensor, memory_config)
    return {
        "regime": "owner_fetch",
        "scheme": p.scheme,
        "tile_row_classes": {
            "full": {"segments": [(0, p.page_bytes)]},
            "partial": {
                "present": p.last_row_is_partial,
                "transfer": p.partial_tile_transfer,
                "valid_rows": p.last_row_valid_rows,
                "segments": list(p.partial_segments),
            },
        },
        "active_cores": [b.core for b in p.blocks],
        "cores": [
            {
                "core": b.core,
                "shard": b.shard,
                "r0": b.r0,
                "c0": b.c0,
                "block_rows": b.block_rows,
                "block_cols": b.block_cols,
                "role": "owner-reader",
            }
            for b in p.blocks
        ],
    }


def create_program_descriptor(p: Plan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    segments = list(p.partial_segments) + [(0, 0)] * (_MAX_PARTIAL_SEGMENTS - len(p.partial_segments))
    reader_ct_args = [
        p.page_bytes,
        p.tensor_cols_tiles,
        p.shard_cols_tiles,
        p.tensor_rows_per_image_tiles,
        int(p.last_row_is_partial),
        len(p.partial_segments),
    ]
    for offset, size in segments:
        reader_ct_args += [offset, size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    for b in p.blocks:
        reader_rt_args[b.core[0]][b.core[1]] = [b.r0, b.c0, b.block_rows, b.block_cols]

    active = _core_ranges([b.core for b in p.blocks])

    reader = ttnn.KernelDescriptor(
        kernel_source=str(READER_KERNEL),
        core_ranges=active,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[])


def set_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor, output_tensor) -> None:
    """Patch the only per-call values of a cached descriptor: the two buffer addresses."""
    reader = descriptor.kernels[0]
    reader.common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
