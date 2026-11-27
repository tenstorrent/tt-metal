# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_fill program descriptor: regime R1, tile row-0 segment scatter.

One writer kernel (BRISC, NoC1) per core. Each core builds a run of
`segment_elems` copies of the fill element in `cb_fill_row`, then writes that
run to row 0 of every tile it owns: one one-packet NoC write per face of row 0
(`segments_per_tile` writes at in-page offsets `s * face_bytes`). There is no
reader and no compute kernel: the input is never read and no arithmetic runs on
the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Blocking-model knobs (op_design.md -> Blocking Model). Each is defined once
# here; every dependent quantity below derives from it. All are part of the
# descriptor cache key.
# ---------------------------------------------------------------------------

# Cap on the cores the `w_tiles` split uses. None = the whole compute grid.
MAX_FILL_CORES = None

# Tiles per block of the per-core schedule. None = the core's whole assignment
# (one block per core).
BLOCK_W_TILES = None

# Pages of `cb_fill_row`. The content is constant and has no producer/consumer
# cycle, so a depth above 1 buys nothing.
FILL_ROW_DEPTH = 1

CB_FILL_ROW = 0

REGIME_TILE = "R1_tile_row0_segments"
REGIME_ROW_MAJOR = "R2_rm_chunks"

_ZONE_BUILD = "G6FILL-BUILD-ROW"
_ZONE_ISSUE = "G6FILL-ISSUE"
_ZONE_DRAIN = "G6FILL-DRAIN"
_ROLE_FILL_WRITER = "fill_writer"

# Smallest NOC_MAX_BURST_SIZE across supported archs (Wormhole's 512-B config).
_ONE_PACKET_MAX_BYTES = 512


@dataclass(frozen=True)
class FillPlan:
    regime: str
    tensor_w_tiles: int
    segment_elems: int
    segment_bytes: int
    segments_per_tile: int
    face_bytes: int
    page_size: int
    block_w_tiles: int
    num_cores: int
    all_cores: object  # ttnn.CoreRangeSet
    # (core, start_tile, core_w_tiles) in tile order
    assignments: tuple

    def report(self) -> dict:
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "core_ranges": str(self.all_cores),
            "tiles_per_core": {f"({c.x},{c.y})": n for c, _, n in self.assignments},
            "block_w_tiles": self.block_w_tiles,
            "segment_bytes": self.segment_bytes,
            "segments_per_tile": self.segments_per_tile,
            "roles": {f"({c.x},{c.y})": _ROLE_FILL_WRITER for c, _, _ in self.assignments},
            "zones": [_ZONE_BUILD, _ZONE_ISSUE, _ZONE_DRAIN],
        }


def select_regime(tensor: ttnn.Tensor) -> str:
    return REGIME_TILE if tensor.layout == ttnn.TILE_LAYOUT else REGIME_ROW_MAJOR


def plan_fill(tensor: ttnn.Tensor) -> FillPlan:
    """The work split and segment geometry. Shared by the descriptor build and the plan report."""
    regime = select_regime(tensor)
    if regime != REGIME_TILE:
        raise NotImplementedError(f"g6_fill: regime {regime} is not built")

    tile_h, tile_w = tensor.tile.tile_shape
    face_h, face_w = tensor.tile.face_shape
    element_size = tensor.element_size()
    # Positions 16..31 of a tile row live in face 1, not after position 15:
    # the segment geometry must come from the real tile/face shape.
    assert (tile_h, tile_w) == (32, 32), f"g6_fill: tile shape {(tile_h, tile_w)} is not (32, 32)"
    assert (face_h, face_w) == (16, 16), f"g6_fill: face shape {(face_h, face_w)} is not (16, 16)"
    assert element_size == 4, f"g6_fill: element size {element_size} is not 4"

    segment_elems = face_w
    segment_bytes = segment_elems * element_size
    segments_per_tile = tile_w // face_w
    face_bytes = face_h * face_w * element_size
    assert segment_bytes <= _ONE_PACKET_MAX_BYTES
    # Blackhole DRAM writes need local L1 offset == NoC offset (mod 16).
    assert face_bytes % 16 == 0 and segment_bytes % 16 == 0

    n = int(tensor.shape[0])
    tensor_w_tiles = tensor.buffer_num_pages()
    assert tensor_w_tiles == -(-n // tile_w), f"g6_fill: {tensor_w_tiles} pages for N={n}"
    assert tensor_w_tiles < 2**32

    grid = tensor.device().compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    max_cores = grid_cores if MAX_FILL_CORES is None else max(1, min(MAX_FILL_CORES, grid_cores))
    candidate_cores = ttnn.num_cores_to_corerangeset(max_cores, grid, True)
    num_cores, all_cores, group_1, group_2, tiles_g1, tiles_g2 = ttnn.split_work_to_cores(
        candidate_cores, tensor_w_tiles, True
    )

    assignments = []
    start_tile = 0
    for group, tiles in ((group_1, tiles_g1), (group_2, tiles_g2)):
        if tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignments.append((core, start_tile, tiles))
            start_tile += tiles
    assert start_tile == tensor_w_tiles and len(assignments) == num_cores

    block_w_tiles = tiles_g1 if BLOCK_W_TILES is None else max(1, min(BLOCK_W_TILES, tiles_g1))

    return FillPlan(
        regime=regime,
        tensor_w_tiles=tensor_w_tiles,
        segment_elems=segment_elems,
        segment_bytes=segment_bytes,
        segments_per_tile=segments_per_tile,
        face_bytes=face_bytes,
        page_size=tensor.buffer_page_size(),
        block_w_tiles=block_w_tiles,
        num_cores=num_cores,
        all_cores=all_cores,
        assignments=tuple(assignments),
    )


# Common runtime arg slots of the writer; patched on every call.
COMMON_ARG_DST_ADDR = 0
COMMON_ARG_FILL_BITS = 1


def create_program_descriptor(output_tensor: ttnn.Tensor, plan: FillPlan):
    """Returns (ProgramDescriptor, writer KernelDescriptor).

    The fill element and the output address are common runtime args, never
    compile-time args or defines, so a cached program takes a new value.
    """
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
        plan.segment_elems,
        plan.segment_bytes,
        plan.segments_per_tile,
        plan.face_bytes,
        plan.block_w_tiles,
        plan.page_size,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for core, start_tile, core_w_tiles in plan.assignments:
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


def descriptor_cache_key(tensor: ttnn.Tensor) -> tuple:
    mc = tensor.memory_config()
    return (
        tensor.device().id(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.layout,
        mc.memory_layout,
        mc.buffer_type,
        MAX_FILL_CORES,
        BLOCK_W_TILES,
        FILL_ROW_DEPTH,
    )
