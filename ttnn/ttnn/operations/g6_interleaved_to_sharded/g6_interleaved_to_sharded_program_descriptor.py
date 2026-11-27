# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_interleaved_to_sharded: owner-fetch on the shard grid.

Each shard's owner core reads its own tile pages from the DRAM-interleaved input
straight into its L1 shard (one NoC read per tile, one barrier per issuer). No CB,
no compute kernel, no writer kernel: every read lands at its final address.

Everything per call except the two buffer addresses is computed once per cache key
and kept as KernelDescriptor objects; a call only patches their common runtime args.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
READER_KERNEL = str(KERNEL_DIR / "g6_interleaved_to_sharded_reader.cpp")

# Module knobs (op_design.md, Parameters). Never call arguments; every value is in the cache key.
KNOBS = {
    # 1: NCRISC/NOC0 issues the whole block. 2: + BRISC/NOC1 takes the second half (perf lamp L1).
    "issuers_per_core": 1,
    # How a tile of an image's last (partial) tile row is read:
    #   "valid_rows": one span from tile byte 0 up to the last logical row (crosses face 0).
    #   "face_rows":  two spans, only the logical rows of the two faces of each face row (perf lamp L2).
    #   "whole_tile": the whole page (A/B reference only).
    "partial_tile_read": "valid_rows",
}
KNOB_VALUES = {
    "issuers_per_core": (1, 2),
    "partial_tile_read": ("valid_rows", "face_rows", "whole_tile"),
}

# Ablation scaffolding: drop the NoC reads, keep launch, addressing and barrier (sync-only floor).
ABLATE_READS = False

# Free L1 per core the model leaves to the op (BB-05); the output shard counts inside it.
L1_BUDGET_BYTES = 1347776

# Issuer i runs on (processor, NoC id). The NoC id is also the kernel's CT arg, so the
# accessor, the reads and the barrier all use the NoC the kernel is configured on.
ISSUERS = (
    (ttnn.DataMovementProcessor.RISCV_1, 0),
    (ttnn.DataMovementProcessor.RISCV_0, 1),
)
_NOC = {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}

WIDTH = "width"
HEIGHT = "height"
_SCHEME = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: WIDTH,
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: HEIGHT,
}


def _ceil_div(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Geometry:
    """Folded tile grid of the input and the shard grid of the target placement."""

    scheme: str
    num_images: int
    image_tile_rows: int
    tensor_tile_rows: int
    tensor_tile_cols: int
    h_rem: int
    tile_bytes: int
    face_h: int
    face_bytes: int
    face_row_bytes: int
    faces_per_tile_row: int
    shard_tile_rows: int
    shard_tile_cols: int
    num_shards: int
    cores: tuple  # G: (x, y) of every grid core in the orientation's order


def compute_geometry(input_tensor, memory_config) -> Geometry:
    """Pure geometry. Assumes a TILE input and a height/width sharded memory_config with a shard spec."""
    shape = list(input_tensor.shape)
    tile_h, tile_w = input_tensor.tile.tile_shape
    face_h, face_w = input_tensor.tile.face_shape
    element_size = input_tensor.element_size()

    H, W = shape[-2], shape[-1]
    num_images = 1
    for dim in shape[:-2]:
        num_images *= dim
    image_tile_rows = _ceil_div(H, tile_h)
    tensor_tile_rows = num_images * image_tile_rows
    tensor_tile_cols = _ceil_div(W, tile_w)

    shard_spec = memory_config.shard_spec
    shard_h, shard_w = shard_spec.shape
    shard_tile_rows = shard_h // tile_h
    shard_tile_cols = shard_w // tile_w
    scheme = _SCHEME[memory_config.memory_layout]
    if scheme == WIDTH:
        num_shards = _ceil_div(tensor_tile_cols, max(shard_tile_cols, 1))
    else:
        num_shards = _ceil_div(tensor_tile_rows, max(shard_tile_rows, 1))

    row_wise = shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = tuple((core.x, core.y) for core in ttnn.corerange_to_cores(shard_spec.grid, None, row_wise))

    face_row_bytes = face_w * element_size
    return Geometry(
        scheme=scheme,
        num_images=num_images,
        image_tile_rows=image_tile_rows,
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        h_rem=H % tile_h,
        tile_bytes=tile_h * tile_w * element_size,
        face_h=face_h,
        face_bytes=face_h * face_row_bytes,
        face_row_bytes=face_row_bytes,
        faces_per_tile_row=tile_w // face_w,
        shard_tile_rows=shard_tile_rows,
        shard_tile_cols=shard_tile_cols,
        num_shards=num_shards,
        cores=cores,
    )


def partial_spans(geom: Geometry, partial_tile_read: str):
    """((0, first_bytes), (second_offset, second_bytes)) read from each tile of an image's last tile row.

    Row k of a tile sits in face row k // face_h, as one face_row_bytes piece in each of that face
    row's faces. Both spans keep their tile-relative offset in the source page and in the shard
    slot, so source and destination stay congruent mod the DRAM read alignment.
    second_bytes == 0 means one span.
    """
    if partial_tile_read == "whole_tile" or geom.h_rem == 0:
        return (0, geom.tile_bytes), (0, 0)
    face_rows_before = (geom.h_rem - 1) // geom.face_h
    rows_in_last_face = geom.h_rem - face_rows_before * geom.face_h
    last_face_row_offset = face_rows_before * geom.faces_per_tile_row * geom.face_bytes
    rows_bytes = rows_in_last_face * geom.face_row_bytes
    if partial_tile_read == "valid_rows":
        whole_faces = face_rows_before * geom.faces_per_tile_row + geom.faces_per_tile_row - 1
        return (0, whole_faces * geom.face_bytes + rows_bytes), (0, 0)
    if geom.faces_per_tile_row != 2:
        raise ValueError("g6_interleaved_to_sharded: face_rows spans need two faces per tile row")
    second_offset = last_face_row_offset + geom.face_bytes
    return (0, last_face_row_offset + rows_bytes), (second_offset, rows_bytes)


def partial_span_bytes(geom: Geometry, partial_tile_read: str) -> int:
    """Total bytes read from each tile of an image's last tile row."""
    (_, first_bytes), (_, second_bytes) = partial_spans(geom, partial_tile_read)
    return first_bytes + second_bytes


def owner_block(geom: Geometry, shard: int):
    """(row_begin, col_begin, block_tile_rows, block_tile_cols) of shard `shard`'s owner, clamped to the grid."""
    if geom.scheme == WIDTH:
        col_begin = shard * geom.shard_tile_cols
        return 0, col_begin, geom.tensor_tile_rows, min(geom.shard_tile_cols, geom.tensor_tile_cols - col_begin)
    row_begin = shard * geom.shard_tile_rows
    return row_begin, 0, min(geom.shard_tile_rows, geom.tensor_tile_rows - row_begin), geom.tensor_tile_cols


def _split(extent: int, parts: int):
    """(begin, size) of `parts` near-equal pieces of [0, extent), larger pieces first."""
    pieces = []
    begin = 0
    for remaining in range(parts, 0, -1):
        size = _ceil_div(extent - begin, remaining)
        pieces.append((begin, size))
        begin += size
    return pieces


def issuer_blocks(block_tile_rows: int, block_tile_cols: int, issuers: int):
    """Per-issuer (row_offset, col_offset, rows, cols) inside an owner's block.

    Cuts block columns; a one-column block is cut along its rows instead. An issuer with
    nothing to do gets an empty rectangle.
    """
    if block_tile_cols > 1:
        return [(0, begin, block_tile_rows, size) for begin, size in _split(block_tile_cols, issuers)]
    return [(begin, 0, size, block_tile_cols) for begin, size in _split(block_tile_rows, issuers)]


@dataclass
class CachedProgram:
    kernels: list  # ttnn.KernelDescriptor, one per issuer; common runtime args patched per call
    plan: dict


_PROGRAM_CACHE: dict = {}


def _grid_key(grid):
    return tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in grid.ranges())


def cache_key(input_tensor, memory_config):
    in_mc = input_tensor.memory_config()
    shard_spec = memory_config.shard_spec
    return (
        id(input_tensor.device()),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        in_mc.memory_layout,
        in_mc.buffer_type,
        memory_config.memory_layout,
        memory_config.buffer_type,
        _grid_key(shard_spec.grid),
        tuple(shard_spec.shape),
        shard_spec.orientation,
        tuple(sorted(KNOBS.items())),
        ABLATE_READS,
    )


def _build(input_tensor, memory_config) -> CachedProgram:
    issuers_per_core = KNOBS["issuers_per_core"]
    partial_tile_read = KNOBS["partial_tile_read"]
    if issuers_per_core not in KNOB_VALUES["issuers_per_core"]:
        raise ValueError(f"g6_interleaved_to_sharded: KNOBS['issuers_per_core']={issuers_per_core!r}")
    if partial_tile_read not in KNOB_VALUES["partial_tile_read"]:
        raise ValueError(f"g6_interleaved_to_sharded: KNOBS['partial_tile_read']={partial_tile_read!r}")

    geom = compute_geometry(input_tensor, memory_config)
    (_, first_span_bytes), (second_span_offset, second_span_bytes) = partial_spans(geom, partial_tile_read)

    issuer_rt_args = [[] for _ in range(issuers_per_core)]
    plan_cores = []
    for shard, (x, y) in enumerate(geom.cores):
        core = ttnn.CoreCoord(x, y)
        if shard < geom.num_shards:
            row_begin, col_begin, block_rows, block_cols = owner_block(geom, shard)
            plan_cores.append(
                {
                    "core": (x, y),
                    "role": "owner",
                    "shard": shard,
                    "block_tile_rows": block_rows,
                    "block_tile_cols": block_cols,
                }
            )
        else:
            row_begin, col_begin, block_rows, block_cols = 0, 0, 0, 0
            plan_cores.append(
                {"core": (x, y), "role": "idle", "shard": None, "block_tile_rows": 0, "block_tile_cols": 0}
            )
        pieces = issuer_blocks(block_rows, block_cols, issuers_per_core)
        for issuer, (dr, dc, rows, cols) in enumerate(pieces):
            issuer_row = row_begin + dr
            issuer_col = col_begin + dc
            # The owner's block starts at its shard's origin, so the slot offset is the in-block offset.
            slot_begin = dr * geom.shard_tile_cols + dc
            issuer_rt_args[issuer].append(
                (
                    core,
                    [
                        issuer_row,
                        issuer_col,
                        rows,
                        cols,
                        slot_begin,
                        issuer_row % geom.image_tile_rows,
                    ],
                )
            )

    accessor_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    defines = [("G6I2S_ABLATE_READS", "1")] if ABLATE_READS else []
    kernels = []
    for issuer in range(issuers_per_core):
        processor, noc_id = ISSUERS[issuer]
        ct_args = [
            geom.tensor_tile_cols,
            geom.shard_tile_cols,
            geom.image_tile_rows,
            geom.tile_bytes,
            first_span_bytes,
            second_span_offset,
            second_span_bytes,
            noc_id,
        ]
        ct_args.extend(accessor_ct_args)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=READER_KERNEL,
                core_ranges=memory_config.shard_spec.grid,
                compile_time_args=ct_args,
                defines=defines,
                runtime_args=issuer_rt_args[issuer],
                common_runtime_args=[0, 0],
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=_NOC[noc_id]),
            )
        )

    plan = {
        "regime": "owner_fetch",
        "num_shards": geom.num_shards,
        "issuers_per_core": issuers_per_core,
        "partial_tile_read": partial_tile_read,
        "full_span_bytes": geom.tile_bytes,
        "partial_span_bytes": first_span_bytes + second_span_bytes,
        "partial_spans": [(0, first_span_bytes)]
        + ([(second_span_offset, second_span_bytes)] if second_span_bytes else []),
        "cores": plan_cores,
    }
    return CachedProgram(kernels=kernels, plan=plan)


def get_program(input_tensor, memory_config) -> CachedProgram:
    key = cache_key(input_tensor, memory_config)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        cached = _build(input_tensor, memory_config)
        _PROGRAM_CACHE[key] = cached
    return cached


def create_program_descriptor(cached: CachedProgram, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Patch this call's buffer addresses into the cached kernels and wrap them in a ProgramDescriptor."""
    output_address = output_tensor.buffer_address()
    # Every read starts at tile byte 0 of a DRAM page; the landing slot must share its DRAM alignment.
    if output_address % ttnn.get_dram_alignment() != 0:
        raise RuntimeError(
            f"g6_interleaved_to_sharded: output shard at 0x{output_address:x} is not "
            f"{ttnn.get_dram_alignment()}-byte aligned"
        )
    addresses = [input_tensor.buffer_address(), output_address]
    for kernel in cached.kernels:
        kernel.common_runtime_args = addresses
    return ttnn.ProgramDescriptor(kernels=cached.kernels, semaphores=[], cbs=[])
