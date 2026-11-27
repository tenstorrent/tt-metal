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

# How a partial tile row (an image's last tile row when H % tile_h != 0) crosses DRAM:
#   "valid_rows":      only the face rows that hold valid data, rounded up to the DRAM read granule
#                      (one read per face pair that holds valid rows).
#   "valid_rows_span": one read from the first to the last valid byte of the page.
#   "whole_page":      the whole page, padding included.
# Default "valid_rows_span": at H = 1 it reads 576 B per page in one NoC issue instead of 2 x 64 B in
# two, and on FLUX 58 (5 pages per core) the per-read issue cost outweighs the extra bytes
# (device kernel 1.05 us against 1.21 us for "valid_rows", 1.68 us for "whole_page").
# Read at call time by plan() and the op's cache key, so a caller can switch it per process.
PARTIAL_TILE_TRANSFER = "valid_rows_span"
_PARTIAL_TILE_TRANSFERS = ("valid_rows", "valid_rows_span", "whole_page")

# Segment reads a core keeps in flight before it waits on a read barrier. 0: every read of the block
# is issued before the core's single final barrier. The reads land in the output shard itself, so
# the depth costs no L1. Read at call time by plan() and the op's cache key.
READS_IN_FLIGHT = 0

# Data-movement RISC and NoC of the reader. The op has no other kernel, so either RISC is free.
#   "ncrisc_noc0": NCRISC on NoC0 (ttnn.ReaderConfigDescriptor()).
#   "brisc_noc0", "brisc_noc1": BRISC, the RISC that receives the launch, on NoC0 or NoC1.
# Default NCRISC/NoC0: on FLUX 58 both RISCs start their kernel ~590 cycles after the launch, BRISC/NoC0
# measured within noise of it and BRISC/NoC1 slower (its DRAM round trip is ~290 cycles longer).
# Read at call time by the builder and the op's cache key.
READER_PROCESSOR = "ncrisc_noc0"
_READER_PROCESSORS = ("ncrisc_noc0", "brisc_noc0", "brisc_noc1")

# Reader kernels per core. 1: one reader (READER_PROCESSOR). 2: NCRISC and BRISC each read one half
# of the core's shard rectangle, both on NoC0 in dynamic-NoC mode, to halve the per-RISC issue time.
# Default 1: on FLUX 58 (5 reads per core) the two RISCs share the NoC0 interface, so each RISC's issue
# time drops only from ~250 to ~200 cycles while the barrier wait grows; no gain was measured.
# Read at call time by the builder and the op's cache key.
READER_SPLIT = 1
_READER_SPLITS = (1, 2)

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
    reads_in_flight: int
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


def plan(
    input_tensor, memory_config, partial_tile_transfer: str | None = None, reads_in_flight: int | None = None
) -> Plan:
    if partial_tile_transfer is None:
        partial_tile_transfer = PARTIAL_TILE_TRANSFER
    if partial_tile_transfer not in _PARTIAL_TILE_TRANSFERS:
        raise ValueError(f"partial_tile_transfer must be one of {_PARTIAL_TILE_TRANSFERS}")
    if reads_in_flight is None:
        reads_in_flight = READS_IN_FLIGHT
    if reads_in_flight < 0:
        raise ValueError("reads_in_flight must be >= 0 (0: the whole block before one barrier)")
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
        partial_tile_transfer != "whole_page"
        and last_row_valid_rows < tile_h
        and _face_geometry_supported(tile, input_tensor.dtype)
    ):
        segments = _valid_rows_segments(
            last_row_valid_rows, tile, input_tensor.element_size(), ttnn.get_dram_alignment()
        )
        if partial_tile_transfer == "valid_rows_span":
            segments = ((segments[0][0], segments[-1][0] + segments[-1][1] - segments[0][0]),)
        transfer = partial_tile_transfer
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
        reads_in_flight=reads_in_flight,
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
        "reads_in_flight": p.reads_in_flight or "block",
        "readers": READER_SPLIT,
        "reader_processor": READER_PROCESSOR if READER_SPLIT == 1 else "ncrisc_noc0+brisc_noc0",
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


def _reader_config(reader_processor: str):
    if reader_processor == "ncrisc_noc0":
        return ttnn.ReaderConfigDescriptor()
    if reader_processor not in _READER_PROCESSORS:
        raise ValueError(f"reader_processor must be one of {_READER_PROCESSORS}")
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.NOC_0 if reader_processor == "brisc_noc0" else ttnn.NOC.NOC_1,
    )


def _split_block(b: CoreBlock, parts: int, shard_cols_tiles: int) -> list:
    """`parts` sub-rectangles of the core's block, as (r0, c0, rows, cols, first shard slot).

    The block is cut along its longer side; a part may be empty when the block has one page."""
    if b.block_cols >= b.block_rows:
        cuts = [b.block_cols * k // parts for k in range(parts + 1)]
        spans = [(b.r0, b.c0 + lo, b.block_rows, hi - lo) for lo, hi in zip(cuts, cuts[1:])]
    else:
        cuts = [b.block_rows * k // parts for k in range(parts + 1)]
        spans = [(b.r0 + lo, b.c0, hi - lo, b.block_cols) for lo, hi in zip(cuts, cuts[1:])]
    return [(r0, c0, rows, cols, (r0 - b.r0) * shard_cols_tiles + (c0 - b.c0)) for r0, c0, rows, cols in spans]


def _reader_configs(reader_split: int, reader_processor: str) -> list:
    if reader_split not in _READER_SPLITS:
        raise ValueError(f"reader_split must be one of {_READER_SPLITS}")
    if reader_split == 1:
        return [_reader_config(reader_processor)]
    return [
        ttnn.DataMovementConfigDescriptor(
            processor=processor, noc=ttnn.NOC.NOC_0, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
        )
        for processor in (ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0)
    ]


def create_program_descriptor(p: Plan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    segments = list(p.partial_segments) + [(0, 0)] * (_MAX_PARTIAL_SEGMENTS - len(p.partial_segments))
    reader_ct_args = [
        p.page_bytes,
        p.tensor_cols_tiles,
        p.shard_cols_tiles,
        p.tensor_rows_per_image_tiles,
        int(p.last_row_is_partial),
        p.reads_in_flight,
        len(p.partial_segments),
    ]
    for offset, size in segments:
        reader_ct_args += [offset, size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    configs = _reader_configs(READER_SPLIT, READER_PROCESSOR)
    rt_args = [ttnn.RuntimeArgs() for _ in configs]
    for b in p.blocks:
        for args, part in zip(rt_args, _split_block(b, len(configs), p.shard_cols_tiles)):
            args[b.core[0]][b.core[1]] = list(part)

    active = _core_ranges([b.core for b in p.blocks])
    readers = [
        ttnn.KernelDescriptor(
            kernel_source=str(READER_KERNEL),
            core_ranges=active,
            compile_time_args=reader_ct_args,
            runtime_args=args,
            common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
            config=config,
        )
        for config, args in zip(configs, rt_args)
    ]
    return ttnn.ProgramDescriptor(kernels=readers, semaphores=[], cbs=[])


def set_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor, output_tensor) -> None:
    """Patch the only per-call values of a cached descriptor: the two buffer addresses."""
    for reader in descriptor.kernels:
        reader.common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
