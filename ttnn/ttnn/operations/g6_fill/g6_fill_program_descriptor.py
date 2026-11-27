# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill program descriptor: plan (regime, core split, block geometry) and the one-writer program.

Regime R1 (TILE): every core of a row-wise split of the tile pages writes the two
row-0 face segments of each of its tiles from one L1 run of the fill element.
Nothing is read; no compute kernel exists.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = KERNEL_DIR / "g6_fill_writer.cpp"

REGIME_TILE = "R1_tile_row0_segments"
REGIME_ROW_MAJOR = "R2_rm_chunks"

# Blocking-model knobs. Each is read at plan time and is part of the descriptor cache key.
# None means "the design's Phase 0 value": the full compute grid / the whole per-core assignment.
MAX_FILL_CORES = None
BLOCK_W_TILES = None
FILL_ROW_DEPTH = 1

CB_FILL_ROW = 0

# Common runtime-arg slots of the writer. Patched per call on the cached descriptor.
COMMON_ARG_DST_ADDR = 0
COMMON_ARG_FILL_BITS = 1
NUM_COMMON_ARGS = 2

_EXPECTED_TILE_SHAPE = (32, 32)
_EXPECTED_FACE_SHAPE = (16, 16)
_ELEMENT_BYTES = 4


def knob_values():
    """The knob tuple that keys the descriptor cache."""
    return (MAX_FILL_CORES, BLOCK_W_TILES, FILL_ROW_DEPTH)


def select_regime(tensor):
    return REGIME_TILE if tensor.layout == ttnn.TILE_LAYOUT else REGIME_ROW_MAJOR


def _tile_geometry(tensor):
    tile_shape = tuple(int(d) for d in tensor.tile.tile_shape)
    face_shape = tuple(int(d) for d in tensor.tile.face_shape)
    element_bytes = int(tensor.element_size())
    assert tile_shape == _EXPECTED_TILE_SHAPE, f"g6_fill: tile shape {tile_shape}, expected {_EXPECTED_TILE_SHAPE}"
    assert face_shape == _EXPECTED_FACE_SHAPE, f"g6_fill: face shape {face_shape}, expected {_EXPECTED_FACE_SHAPE}"
    assert element_bytes == _ELEMENT_BYTES, f"g6_fill: element size {element_bytes}, expected {_ELEMENT_BYTES}"
    tile_w = tile_shape[1]
    face_h, face_w = face_shape
    return {
        "tile_w": tile_w,
        "segments_per_tile": tile_w // face_w,
        "segment_bytes": face_w * element_bytes,
        # Face f of row 0 starts at f * face_bytes inside the page.
        "segment_stride_bytes": face_h * face_w * element_bytes,
    }


def _fill_cores(grid, max_fill_cores):
    """The core set the split may use: the whole grid, or its first max_fill_cores cores row-wise."""
    grid_cores = grid.x * grid.y
    if max_fill_cores is None or max_fill_cores >= grid_cores:
        return grid
    assert max_fill_cores >= 1, f"g6_fill: MAX_FILL_CORES must be >= 1, got {max_fill_cores}"
    return ttnn.num_cores_to_corerangeset(max_fill_cores, grid, True)


def compute_plan(tensor, *, max_fill_cores=None, block_w_tiles=None, fill_row_depth=1):
    """The plan a call with this tensor dispatches. Shared by the descriptor build and g6_fill_plan()."""
    regime = select_regime(tensor)
    if regime != REGIME_TILE:
        raise NotImplementedError(f"g6_fill: regime {regime} is not built")

    geometry = _tile_geometry(tensor)
    n = int(tensor.shape[-1])
    tensor_w_tiles = int(tensor.buffer_num_pages())
    expected_w_tiles = -(-n // geometry["tile_w"])
    assert tensor_w_tiles == expected_w_tiles, f"g6_fill: {tensor_w_tiles} pages for N={n}, expected {expected_w_tiles}"

    assert tensor_w_tiles < 2**32

    grid = tensor.device().compute_with_storage_grid_size()
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tiles_per_core_1,
        tiles_per_core_2,
    ) = ttnn.split_work_to_cores(_fill_cores(grid, max_fill_cores), tensor_w_tiles, row_wise=True)

    assignment = []  # (core, start_tile, core_w_tiles), start_tile contiguous in row-wise core order
    start_tile = 0
    for group, tiles_per_core in ((core_group_1, tiles_per_core_1), (core_group_2, tiles_per_core_2)):
        if tiles_per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start_tile, tiles_per_core))
            start_tile += tiles_per_core
    assert start_tile == tensor_w_tiles

    max_core_w_tiles = max(tiles_per_core_1, tiles_per_core_2)
    if block_w_tiles is None:
        block = max_core_w_tiles
    else:
        assert block_w_tiles >= 1, f"g6_fill: BLOCK_W_TILES must be >= 1, got {block_w_tiles}"
        block = min(block_w_tiles, max_core_w_tiles)

    assert fill_row_depth >= 1

    return {
        "regime": regime,
        "num_cores": num_cores,
        "all_cores": all_cores,
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in all_cores.ranges()],
        "assignment": assignment,
        "tiles_per_core": {(c.x, c.y): count for c, _, count in assignment},
        "tensor_w_tiles": tensor_w_tiles,
        "block_w_tiles": block,
        "fill_row_depth": fill_row_depth,
        "segment_bytes": geometry["segment_bytes"],
        "segments_per_tile": geometry["segments_per_tile"],
        "segment_stride_bytes": geometry["segment_stride_bytes"],
        "roles": {(c.x, c.y): "fill_writer" for c, _, _ in assignment},
    }


def create_program_descriptor(output_tensor, plan):
    """Build the cached program. Returns (ProgramDescriptor, writer KernelDescriptor).

    The writer's common runtime args (dst_addr, fill_bits) are placeholders here; the caller
    patches them on every call and reassigns ProgramDescriptor.kernels, because .kernels is a
    by-value property and an in-place edit of desc.kernels[0] mutates a temporary copy.
    """
    all_cores = plan["all_cores"]
    segment_bytes = plan["segment_bytes"]

    cb_fill_row = ttnn.CBDescriptor(
        total_size=plan["fill_row_depth"] * segment_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_FILL_ROW,
                data_format=output_tensor.dtype,
                page_size=segment_bytes,
            )
        ],
    )

    writer_ct_args = [
        CB_FILL_ROW,
        segment_bytes,
        plan["segments_per_tile"],
        plan["segment_stride_bytes"],
        plan["block_w_tiles"],
        output_tensor.buffer_aligned_page_size(),
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for core, start_tile, core_w_tiles in plan["assignment"]:
        writer_rt_args[core.x][core.y] = [start_tile, core_w_tiles]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(WRITER_KERNEL),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[0] * NUM_COMMON_ARGS,
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_fill_row])
    return descriptor, writer
