# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill program: regime R1 (tile row-0 segment scatter), one writer kernel per core.

The plan (core split, per-core tile ranges, segment geometry) is built by `build_plan`, and both
the program descriptor and `g6_fill_plan` read it, so the reported plan is the dispatched one.
The descriptor is built once per cache key; a call only patches the two common runtime args.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_SOURCE = str(KERNEL_DIR / "g6_fill_writer.cpp")

# Blocking-model knobs (op_design.md, Blocking Model). Read at call time; both are in the cache key.
# MAX_FILL_CORES: cap on the cores the w_tiles split may use; None is the whole compute grid.
MAX_FILL_CORES = None
# BLOCK_W_TILES: tiles per scatter block; None makes a core's whole assignment one block.
BLOCK_W_TILES = None
# FILL_ROW_DEPTH: pages of cb_fill_row. The run is constant and never refilled, so 1 is the live set.
FILL_ROW_DEPTH = 1

CB_FILL_ROW = 0

# Runtime-arg layouts; the writer kernel reads the same indices.
COMMON_ARG_DST_ADDR = 0
COMMON_ARG_FILL_BITS = 1
NUM_COMMON_ARGS = 2
UNIQUE_ARGS = ("start_tile", "core_w_tiles", "block_w_tiles")

REGIME_TILE = "R1_tile_row0_segments"
REGIME_ROW_MAJOR = "R2_rm_chunks"

# The R1 segment geometry assumes this tile; other tiles would put positions 16-31 in the wrong face.
EXPECTED_TILE_SHAPE = (32, 32)
EXPECTED_FACE_SHAPE = (16, 16)
ELEMENT_BYTES = 4


def select_regime(layout) -> str:
    return REGIME_TILE if layout == ttnn.TILE_LAYOUT else REGIME_ROW_MAJOR


def _segment_geometry(tensor) -> dict:
    tile_shape = tuple(tensor.tile.tile_shape)
    face_shape = tuple(tensor.tile.face_shape)
    element_bytes = tensor.element_size()
    assert tile_shape == EXPECTED_TILE_SHAPE, f"g6_fill: tile {tile_shape}, expected {EXPECTED_TILE_SHAPE}"
    assert face_shape == EXPECTED_FACE_SHAPE, f"g6_fill: face {face_shape}, expected {EXPECTED_FACE_SHAPE}"
    assert element_bytes == ELEMENT_BYTES, f"g6_fill: element size {element_bytes}, expected {ELEMENT_BYTES}"
    face_h, face_w = face_shape
    segment_bytes = face_w * element_bytes
    face_bytes = face_h * face_w * element_bytes
    return {
        "segment_elems": face_w,
        "segment_bytes": segment_bytes,
        "segments_per_tile": tile_shape[1] // face_w,
        "face_bytes": face_bytes,
    }


def _core_cap(grid, max_fill_cores) -> int:
    grid_cores = grid.x * grid.y
    if max_fill_cores is None:
        return grid_cores
    assert max_fill_cores >= 1, f"g6_fill: MAX_FILL_CORES={max_fill_cores} must be >= 1"
    return min(max_fill_cores, grid_cores)


def build_plan(tensor, *, max_fill_cores=None, block_w_tiles=None) -> dict:
    """The R1 work split: contiguous tile ranges over min(tensor_w_tiles, cap) cores, row-wise."""
    regime = select_regime(tensor.layout)
    if regime != REGIME_TILE:
        raise NotImplementedError(f"g6_fill: regime {regime} is not built")

    n = int(tensor.shape[0])
    tensor_w_tiles = tensor.buffer_num_pages()
    assert tensor_w_tiles == -(
        -n // EXPECTED_TILE_SHAPE[1]
    ), f"g6_fill: {tensor_w_tiles} pages for N={n}, expected one tile per 32 positions"
    assert tensor_w_tiles < 2**32, "g6_fill: page id must fit the kernel's uint32 page index"
    assert block_w_tiles is None or block_w_tiles >= 1, f"g6_fill: BLOCK_W_TILES={block_w_tiles} must be >= 1"

    grid = tensor.device().compute_with_storage_grid_size()
    candidate_cores = ttnn.num_cores_to_corerangeset(_core_cap(grid, max_fill_cores), grid, True)
    num_cores, all_cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(
        candidate_cores, tensor_w_tiles, row_wise=True
    )

    assignment = []
    start_tile = 0
    for group, core_w_tiles in ((group_1, tiles_1), (group_2, tiles_2)):
        if core_w_tiles == 0:
            continue
        core_block = core_w_tiles if block_w_tiles is None else min(block_w_tiles, core_w_tiles)
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            assignment.append((core, start_tile, core_w_tiles, core_block))
            start_tile += core_w_tiles
    assert start_tile == tensor_w_tiles and len(assignment) == num_cores, "g6_fill: tile split does not cover N"

    return {
        "regime": regime,
        "tensor_w_tiles": tensor_w_tiles,
        "num_cores": num_cores,
        "all_cores": all_cores,
        "assignment": assignment,
        **_segment_geometry(tensor),
    }


def create_program_descriptor(output_tensor, plan) -> tuple[ttnn.ProgramDescriptor, ttnn.KernelDescriptor]:
    """Descriptor plus its writer KernelDescriptor, whose common args the caller patches per call.

    ProgramDescriptor.kernels is returned by value, so a patch must go through the kept
    KernelDescriptor and be reassigned to the descriptor.
    """
    all_cores = plan["all_cores"]

    cb_fill_row = ttnn.CBDescriptor(
        total_size=FILL_ROW_DEPTH * plan["segment_bytes"],
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_FILL_ROW,
                data_format=output_tensor.dtype,
                page_size=plan["segment_bytes"],
            )
        ],
    )

    dst_accessor = ttnn.TensorAccessorArgs(output_tensor)
    # The kernel's common args start at index 0, so the accessor must not claim any.
    assert not dst_accessor.get_common_runtime_args(), "g6_fill: output accessor needs common runtime args"
    compile_time_args = [CB_FILL_ROW, plan["segment_elems"], plan["segments_per_tile"], plan["face_bytes"]]
    compile_time_args.extend(dst_accessor.get_compile_time_args())

    runtime_args = ttnn.RuntimeArgs()
    for core, start_tile, core_w_tiles, block_w_tiles in plan["assignment"]:
        runtime_args[core.x][core.y] = [start_tile, core_w_tiles, block_w_tiles]

    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER_SOURCE,
        core_ranges=all_cores,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        # Plain uint32 values, never Buffer bindings: on a cache hit generic_op then re-applies every
        # arg from the passed descriptor, so a new fill_value reaches the cached program.
        common_runtime_args=[0] * NUM_COMMON_ARGS,
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_fill_row])
    return descriptor, writer


_PROGRAM_CACHE: dict = {}


def _cache_key(tensor, grid, max_fill_cores, block_w_tiles):
    return (
        tensor.device().id(),
        (grid.x, grid.y),
        tuple(int(d) for d in tensor.shape),
        tensor.dtype,
        tensor.layout,
        tensor.memory_config(),
        max_fill_cores,
        block_w_tiles,
    )


def get_program(output_tensor):
    max_fill_cores, block_w_tiles = MAX_FILL_CORES, BLOCK_W_TILES
    grid = output_tensor.device().compute_with_storage_grid_size()
    key = _cache_key(output_tensor, grid, max_fill_cores, block_w_tiles)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        plan = build_plan(output_tensor, max_fill_cores=max_fill_cores, block_w_tiles=block_w_tiles)
        entry = create_program_descriptor(output_tensor, plan)
        _PROGRAM_CACHE[key] = entry
    return entry


def dispatch(output_tensor, fill_bits: int) -> None:
    """One generic_op: the cached program with this call's address and fill pattern."""
    descriptor, writer = get_program(output_tensor)
    common_args = [0] * NUM_COMMON_ARGS
    common_args[COMMON_ARG_DST_ADDR] = output_tensor.buffer_address()
    common_args[COMMON_ARG_FILL_BITS] = fill_bits
    writer.common_runtime_args = common_args
    descriptor.kernels = [writer]
    # The output is the only io tensor: the input is never read, so it is never an operand.
    ttnn.generic_op([output_tensor], descriptor)
