# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill program descriptor: one writer kernel, regime R1 (tile row-0 segment scatter).

The program has no reader and no compute kernel: nothing is read and nothing is computed
on device. Each core writes the two row-0 segments (face 0 and face 1) of the tiles it owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME_TILE_ROW0 = "R1_tile_row0_segments"
REGIME_RM_CHUNKS = "R2_rm_chunks"

# Block-model knobs (op_design.md, Blocking Model). Each is part of the descriptor cache key.
# MAX_FILL_CORES caps the w_tiles split; None spreads it over the whole compute grid.
MAX_FILL_CORES = None
# BLOCK_W_TILES caps the tiles of one scatter block; None makes the block the core's whole
# assignment, so every core runs exactly one block.
BLOCK_W_TILES = None
# cb_fill_row holds a constant read by every write: one page is the whole live set.
FILL_ROW_DEPTH = 1

CB_FILL_ROW = 0

# Common runtime-arg slots of the writer, patched on every call.
COMMON_ARG_DST_ADDR = 0
COMMON_ARG_FILL_BITS = 1

_TILE_SHAPE = (32, 32)
_FACE_SHAPE = (16, 16)
_ELEMENT_BYTES = 4


@dataclass(frozen=True)
class FillPlan:
    regime: str
    tensor_w_tiles: int
    num_cores: int
    all_cores: ttnn.CoreRangeSet
    core_groups: tuple  # ((CoreRangeSet, tiles_per_core), ...) in start-tile order
    block_w_tiles: int
    segment_bytes: int
    segments_per_tile: int
    segment_stride_bytes: int
    page_bytes: int


def regime_of(tensor) -> str:
    return REGIME_TILE_ROW0 if tensor.layout == ttnn.TILE_LAYOUT else REGIME_RM_CHUNKS


def make_plan(output_tensor, *, max_fill_cores=None, block_w_tiles=None) -> FillPlan:
    """The core plan of one call; the descriptor build and g6_fill_plan both read it."""
    regime = regime_of(output_tensor)
    if regime != REGIME_TILE_ROW0:
        raise NotImplementedError(f"g6_fill: regime {regime} is not built")

    tile_shape = tuple(output_tensor.tile.tile_shape)
    face_shape = tuple(output_tensor.tile.face_shape)
    element_bytes = output_tensor.element_size()
    assert tile_shape == _TILE_SHAPE, f"g6_fill: tile shape {tile_shape}, expected {_TILE_SHAPE}"
    assert face_shape == _FACE_SHAPE, f"g6_fill: face shape {face_shape}, expected {_FACE_SHAPE}"
    assert element_bytes == _ELEMENT_BYTES, f"g6_fill: element size {element_bytes}, expected {_ELEMENT_BYTES}"

    face_h, face_w = face_shape
    tile_w = tile_shape[1]
    segment_bytes = face_w * element_bytes
    segments_per_tile = tile_w // face_w
    # Positions 16..31 of row 0 are row 0 of face 1, one whole face after positions 0..15.
    segment_stride_bytes = face_h * face_w * element_bytes

    n = int(output_tensor.shape[-1])
    tensor_w_tiles = output_tensor.buffer_num_pages()
    assert tensor_w_tiles == -(-n // tile_w), f"g6_fill: {tensor_w_tiles} pages for N={n}"
    assert tensor_w_tiles < 2**32

    grid = output_tensor.device().compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    cap = grid_cores if max_fill_cores is None else max(1, min(int(max_fill_cores), grid_cores))
    candidate_cores = ttnn.num_cores_to_corerangeset(cap, grid, True)
    num_cores, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        candidate_cores, tensor_w_tiles, row_wise=True
    )
    core_groups = tuple(
        (group, per_core) for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)) if per_core > 0
    )

    block = per_core_1 if block_w_tiles is None else max(1, min(int(block_w_tiles), per_core_1))

    return FillPlan(
        regime=regime,
        tensor_w_tiles=tensor_w_tiles,
        num_cores=num_cores,
        all_cores=all_cores,
        core_groups=core_groups,
        block_w_tiles=block,
        segment_bytes=segment_bytes,
        segments_per_tile=segments_per_tile,
        segment_stride_bytes=segment_stride_bytes,
        page_bytes=output_tensor.buffer_page_size(),
    )


def core_assignment(plan: FillPlan):
    """[(core, start_tile, core_w_tiles)] in row-wise core order."""
    assignment = []
    start_tile = 0
    for group, per_core in plan.core_groups:
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start_tile, per_core))
            start_tile += per_core
    assert start_tile == plan.tensor_w_tiles, f"g6_fill: split covers {start_tile} of {plan.tensor_w_tiles} tiles"
    return assignment


def create_program_descriptor(output_tensor, plan: FillPlan):
    """Returns (ProgramDescriptor, writer KernelDescriptor); the caller patches the common args."""
    cb_fill_row = ttnn.CBDescriptor(
        total_size=FILL_ROW_DEPTH * plan.segment_bytes,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_FILL_ROW,
                data_format=output_tensor.dtype,
                page_size=plan.segment_bytes,
            )
        ],
    )

    writer_ct_args = [
        CB_FILL_ROW,
        plan.segment_bytes,
        plan.segments_per_tile,
        plan.segment_stride_bytes,
        plan.page_bytes,
        plan.block_w_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for core, start_tile, core_w_tiles in core_assignment(plan):
        writer_rt_args[core.x][core.y] = [start_tile, core_w_tiles]

    common_args = [0, 0]
    common_args[COMMON_ARG_DST_ADDR] = output_tensor.buffer_address()

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_fill_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=common_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_fill_row])
    return descriptor, writer


def set_call_args(descriptor, writer, dst_addr: int, fill_bits: int) -> None:
    """Patch the per-call values into the cached descriptor.

    ProgramDescriptor.kernels is a by-value property, so the patched KernelDescriptor has to be
    assigned back; mutating descriptor.kernels[0] would edit a temporary copy.
    """
    args = [0, 0]
    args[COMMON_ARG_DST_ADDR] = dst_addr
    args[COMMON_ARG_FILL_BITS] = fill_bits
    writer.common_runtime_args = args
    descriptor.kernels = [writer]
