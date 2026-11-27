# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute program descriptor: regime R1 `rows_tile` (dims (0, 2, 1, 3), TILE).

Plan selection (`select_plan`), per-core block ranges, CBs and kernels. Every knob of the
Blocking Model in op_design.md is defined once here; CB page counts, kernel loop bounds and
the core split derive from it.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME_ROWS_TILE = "rows_tile"

# Blocking-model knobs (op_design.md, Axes and Buffer-depth knobs).
BLOCK_WT = 1
MAX_DEPTH_IN = 2
MAX_DEPTH_STAGING = 2

CB_INPUT_TILES = 0
CB_OUTPUT_STAGING = 1

_ROLE = "rows_tile reader+writer"


@dataclass(frozen=True)
class Geometry:
    """Tile and face geometry of one dtype, in rows and bytes."""

    tile_h: int
    tile_w: int
    face_h: int
    face_w: int
    tile_bytes: int
    face_bytes: int
    face_row_bytes: int

    @property
    def faces_per_band(self):
        return self.tile_w // self.face_w


def tile_geometry(tensor) -> Geometry:
    tile_h, tile_w = (int(v) for v in tensor.tile.tile_shape)
    face_h, face_w = (int(v) for v in tensor.tile.face_shape)
    # Byte offsets of face rows assume the 32x32 tile of 16x16 faces (op_design.md, Mechanism caps).
    assert (tile_h, tile_w) == (32, 32), f"g6_permute: tile shape {(tile_h, tile_w)} is not 32x32"
    assert (face_h, face_w) == (16, 16), f"g6_permute: face shape {(face_h, face_w)} is not 16x16"
    element_bytes = int(tensor.element_size())
    tile_bytes = int(tensor.tile.get_tile_size(tensor.dtype))
    assert tile_bytes == tile_h * tile_w * element_bytes, "g6_permute: tile bytes do not match an unpacked tile"
    return Geometry(
        tile_h=tile_h,
        tile_w=tile_w,
        face_h=face_h,
        face_w=face_w,
        tile_bytes=tile_bytes,
        face_bytes=face_h * face_w * element_bytes,
        face_row_bytes=face_w * element_bytes,
    )


def _div_up(x, y):
    return -(-x // y)


def _block_extents(tile_h):
    """Legal block_a / block_b values: the tile height halved down to 1."""
    extents = []
    e = tile_h
    while e >= 1:
        extents.append(e)
        e //= 2
    return extents


def num_blocks_of(n, a, b, wt, block_a, block_b, block_wt):
    return n * _div_up(a, block_a) * _div_up(b, block_b) * _div_up(wt, block_wt)


def max_cores_for(device):
    """The core-count knob: the whole worker grid (op_design.md, Work Distribution)."""
    grid = device.compute_with_storage_grid_size()
    return int(grid.x) * int(grid.y)


def select_plan(n, a, b, wt, max_cores, tile_h, block_wt):
    """The occupancy walk of op_design.md: closed cells, then halve a, then halve b.

    Each step is taken while its block count fits `max_cores`; the walk stops at the first step that
    does not. Returns (block_a, block_b).
    """
    extents = _block_extents(tile_h)
    walk = [(e, tile_h) for e in extents] + [(1, e) for e in extents[1:]]
    chosen = walk[0]
    for step in walk[1:]:
        if num_blocks_of(n, a, b, wt, *step, block_wt) > max_cores:
            break
        chosen = step
    return chosen


def _split(grid, num_blocks, max_cores):
    """Contiguous block ranges on min(num_blocks, max_cores) cores, row-wise (one or two rectangles)."""
    cores = ttnn.num_cores_to_corerangeset(min(num_blocks, max_cores), grid, True)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(cores, num_blocks, row_wise=True)
    assignment = []
    start = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append(((int(core.x), int(core.y)), start, int(per_core)))
            start += per_core
    assert start == num_blocks, f"g6_permute: split covers {start} of {num_blocks} blocks"
    return all_cores, assignment


def plan_key(input_tensor, canonical_dims):
    """Everything that changes the program: shapes, dims, dtype, layout, placement, grid, core knob."""
    memory_config = input_tensor.memory_config()
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        "g6_permute",
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        tuple(canonical_dims),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        str(memory_config.buffer_type),
        str(memory_config.memory_layout),
        (int(grid.x), int(grid.y)),
        max_cores_for(device),
        BLOCK_WT,
        MAX_DEPTH_IN,
        MAX_DEPTH_STAGING,
    )


def _program_hash(key, plan):
    knobs = tuple(plan[k] for k in ("block_a", "block_b", "block_wt", "depth_in", "depth_staging"))
    digest = hashlib.blake2b(repr((key, knobs)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build_plan(input_tensor):
    """Host-only plan of the rows_tile regime for a validated input; no allocation, no dispatch."""
    geometry = tile_geometry(input_tensor)
    n, a, b, w = (int(d) for d in input_tensor.shape)
    wt = _div_up(w, geometry.tile_w)
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    max_cores = max_cores_for(device)
    block_wt = BLOCK_WT

    block_a, block_b = select_plan(n, a, b, wt, max_cores, geometry.tile_h, block_wt)
    num_blocks = num_blocks_of(n, a, b, wt, block_a, block_b, block_wt)
    all_cores, assignment = _split(grid, num_blocks, max_cores)
    max_blocks_per_core = max(count for _, _, count in assignment)
    regroup = block_a > 1
    return {
        "regime": REGIME_ROWS_TILE,
        "block_a": block_a,
        "block_b": block_b,
        "block_wt": block_wt,
        "num_blocks": num_blocks,
        "num_cores": len(assignment),
        "depth_in": min(MAX_DEPTH_IN, max_blocks_per_core),
        "depth_staging": min(MAX_DEPTH_STAGING, max_blocks_per_core) if regroup else 0,
        "regroup": regroup,
        "cores": [
            {"core": core, "role": _ROLE, "start_block": start, "num_blocks": count}
            for core, start, count in assignment
        ],
        "_all_cores": all_cores,
    }


def public_plan(plan):
    return copy.deepcopy({k: v for k, v in plan.items() if not k.startswith("_")})


def create_program_descriptor(input_tensor, output_tensor, plan, key):
    geometry = tile_geometry(input_tensor)
    n, a, b, w = (int(d) for d in input_tensor.shape)
    wt = _div_up(w, geometry.tile_w)
    in_padded = [int(d) for d in input_tensor.padded_shape]
    out_padded = [int(d) for d in output_tensor.padded_shape]
    assert input_tensor.buffer_page_size() == geometry.tile_bytes
    assert output_tensor.buffer_page_size() == geometry.tile_bytes

    block_a, block_b, block_wt = plan["block_a"], plan["block_b"], plan["block_wt"]
    all_cores = plan["_all_cores"]

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan["depth_in"] * block_a * block_wt * geometry.tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=geometry.tile_bytes
                )
            ],
        )
    ]
    if plan["regroup"]:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan["depth_staging"] * block_b * block_wt * geometry.tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_OUTPUT_STAGING, data_format=output_tensor.dtype, page_size=geometry.tile_bytes
                    )
                ],
            )
        )

    # Same order as the constants of kernels/g6_permute_common.hpp.
    shared_ct_args = [
        a,
        b,
        wt,
        in_padded[1],
        in_padded[2] // geometry.tile_h,
        in_padded[3] // geometry.tile_w,
        out_padded[1],
        out_padded[2] // geometry.tile_h,
        out_padded[3] // geometry.tile_w,
        block_a,
        block_b,
        block_wt,
        geometry.tile_h,
        geometry.face_h,
        geometry.faces_per_band,
        geometry.tile_bytes,
        geometry.face_bytes,
        geometry.face_row_bytes,
        CB_INPUT_TILES,
    ]
    reader_ct_args = list(shared_ct_args)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = list(shared_ct_args) + [CB_OUTPUT_STAGING, plan["depth_staging"]]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    per_core_rt_args = ttnn.RuntimeArgs()
    for core in plan["cores"]:
        x, y = core["core"]
        per_core_rt_args[x][y] = [core["start_block"], core["num_blocks"]]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=per_core_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=per_core_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    descriptor.custom_program_hash = _program_hash(key, plan)
    return descriptor


READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1


def patch_addresses(descriptor, input_tensor, output_tensor):
    """Cache hit: only the two buffer addresses change."""
    kernels = descriptor.kernels
    kernels[READER_KERNEL_INDEX].common_runtime_args[0] = input_tensor.buffer_address()
    kernels[WRITER_KERNEL_INDEX].common_runtime_args[0] = output_tensor.buffer_address()
