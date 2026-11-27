# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill program descriptor: one writer kernel per core, regime R1 (tile row-0 segment scatter).

The plan (`make_plan`) is the single source of the work split; the descriptor and the
public `g6_fill_plan` report both read it, so the report cannot drift from the dispatch.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME_TILE = "R1_tile_row0_segments"
REGIME_ROW_MAJOR = "R2_rm_chunks"

# Block-model knobs (op_design.md -> Blocking Model). Each is defined here once.
# Cores the w_tiles split may use; None = the whole compute grid.
MAX_FILL_CORES = None
# Tiles per scatter block; None = the core's whole assignment (one block per core).
BLOCK_W_TILES = None
# Pages of cb_fill_row. The row is constant and never refilled, so 1 is all it can use.
FILL_ROW_DEPTH = 1

CB_FILL_ROW = 0

# Writer common runtime-arg slots; patched on every call, never hashed.
CRTA_DST_ADDR = 0
CRTA_FILL_BITS = 1
NUM_COMMON_RT_ARGS = 2


def select_regime(layout):
    return REGIME_TILE if layout == ttnn.TILE_LAYOUT else REGIME_ROW_MAJOR


def _tile_geometry(tensor):
    tile_h, tile_w = tensor.tile.tile_shape
    face_h, face_w = tensor.tile.face_shape
    element_size = tensor.element_size()
    assert (tile_h, tile_w) == (32, 32), f"g6_fill: tile shape {(tile_h, tile_w)}, expected (32, 32)"
    assert (face_h, face_w) == (16, 16), f"g6_fill: face shape {(face_h, face_w)}, expected (16, 16)"
    assert element_size == 4, f"g6_fill: element size {element_size}, expected 4"
    return {
        "segments_per_tile": tile_w // face_w,
        "segment_bytes": face_w * element_size,
        "face_bytes": face_h * face_w * element_size,
    }


def _fill_grid(device):
    grid_size = device.compute_with_storage_grid_size()
    grid_cores = grid_size.x * grid_size.y
    if MAX_FILL_CORES is None or MAX_FILL_CORES >= grid_cores:
        return grid_size
    assert MAX_FILL_CORES >= 1, f"g6_fill: MAX_FILL_CORES={MAX_FILL_CORES} must be >= 1"
    return ttnn.num_cores_to_corerangeset(MAX_FILL_CORES, grid_size, True)


def make_plan(tensor):
    """Work split and geometry of one call; depends only on the cache-key fields."""
    regime = select_regime(tensor.layout)
    if regime != REGIME_TILE:
        raise NotImplementedError(f"g6_fill: regime {regime} is not built")

    n = int(tensor.shape[-1])
    tensor_w_tiles = tensor.buffer_num_pages()
    assert tensor_w_tiles == -(-n // 32), f"g6_fill: {tensor_w_tiles} pages for N={n}"
    assert tensor_w_tiles < 2**32

    num_cores, all_cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(
        _fill_grid(tensor.device()), tensor_w_tiles, row_wise=True
    )
    assignment = []
    start_tile = 0
    for group, tiles in ((group_1, tiles_1), (group_2, tiles_2)):
        if tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start_tile, tiles))
            start_tile += tiles
    assert start_tile == tensor_w_tiles and len(assignment) == num_cores

    block_w_tiles = tiles_1 if BLOCK_W_TILES is None else min(BLOCK_W_TILES, tiles_1)
    assert block_w_tiles >= 1

    return {
        "regime": regime,
        "tensor_w_tiles": tensor_w_tiles,
        "num_cores": num_cores,
        "all_cores": all_cores,
        "assignment": assignment,
        "block_w_tiles": block_w_tiles,
        **_tile_geometry(tensor),
    }


def create_program_descriptor(output_tensor):
    """Build the cached descriptor. Returns (ProgramDescriptor, writer KernelDescriptor).

    The common runtime args are placeholders; the caller patches dst_addr and fill_bits
    on every call and reassigns `desc.kernels`.
    """
    plan = make_plan(output_tensor)
    all_cores = plan["all_cores"]
    segment_bytes = plan["segment_bytes"]

    cb_fill_row = ttnn.CBDescriptor(
        total_size=FILL_ROW_DEPTH * segment_bytes,
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
        plan["face_bytes"],
        plan["segments_per_tile"],
        plan["block_w_tiles"],
        output_tensor.buffer_aligned_page_size(),
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for core, start_tile, core_w_tiles in plan["assignment"]:
        writer_rt_args[core.x][core.y] = [start_tile, core_w_tiles]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_fill_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[0] * NUM_COMMON_RT_ARGS,
        config=ttnn.WriterConfigDescriptor(),
    )

    desc = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[cb_fill_row])
    return desc, writer


def plan_report(tensor):
    """The plan a call on `tensor` selects, in reportable form (BB-08, BB-21)."""
    plan = make_plan(tensor)
    return {
        "regime": plan["regime"],
        "num_cores": plan["num_cores"],
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in plan["all_cores"].ranges()],
        "tiles_per_core": {(c.x, c.y): n for c, _, n in plan["assignment"]},
        "block_w_tiles": plan["block_w_tiles"],
        "segment_bytes": plan["segment_bytes"],
        "segments_per_tile": plan["segments_per_tile"],
        "roles": {(c.x, c.y): "fill_writer" for c, _, _ in plan["assignment"]},
    }
