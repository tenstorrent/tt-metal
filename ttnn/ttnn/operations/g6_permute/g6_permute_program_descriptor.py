# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regime rows_tile: plan (unit split, block knobs) and ProgramDescriptor.

Work unit: one output tile (n, b, ta, tw), ordered b-fastest:
    u = ((n * tensor_a_tiles + ta) * tensor_w_tiles + tw) * B + b
Each core owns a contiguous unit range and cuts it into blocks by the block rule
(`_block_extent`, mirrored in kernels/g6_permute_common.hpp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_ROWS = 32
FACE_COLS = 16

CB_INPUT_ROWS = 0
CB_OUTPUT_STAGING = 1

# Block knobs whose Phase 0 value is fixed by the design. They are realized by the block rule
# ("keep n", "keep tw"); a refinement that turns one must teach the block rule the new extent.
BLOCK_N = 1
BLOCK_W_TILES = 1
MAX_DEPTH_IN = 2
DEPTH_OUT = 1


@dataclass
class Plan:
    regime: str
    num_cores: int
    total_units: int
    units_per_core_g1: int
    units_per_core_g2: int
    a_tile_rows: int
    block_b_rows: int
    depth_in: int
    max_blocks_per_core: int
    core_roles: dict
    # Geometry and per-core assignment the descriptor is built from.
    shape: tuple = ()
    tensor_a_tiles: int = 0
    tensor_b_tiles: int = 0
    tensor_w_tiles: int = 0
    all_cores: object = None
    assignment: list = field(default_factory=list)  # [(CoreCoord, start_unit, num_units)]


def _div_up(a, b):
    return (a + b - 1) // b


def _block_extent(b, remaining, B, block_b_rows):
    """Units in the block that starts at row b: stays in range, in B, in one b-tile, <= block_b_rows."""
    return min(block_b_rows, remaining, B - b, TILE_ROWS - b % TILE_ROWS)


def _count_blocks(start, count, B, block_b_rows):
    blocks = 0
    b = start % B
    while count > 0:
        run = _block_extent(b, count, B, block_b_rows)
        count -= run
        b = (b + run) % B
        blocks += 1
    return blocks


def build_plan(input_tensor, canonical_dims) -> Plan:
    assert tuple(canonical_dims) == (0, 2, 1, 3), "only regime rows_tile is built"
    N, A, B, W = (int(d) for d in input_tensor.shape)
    tensor_a_tiles = _div_up(A, TILE_ROWS)
    tensor_b_tiles = _div_up(B, TILE_ROWS)
    tensor_w_tiles = _div_up(W, TILE_ROWS)

    a_tile_rows = min(TILE_ROWS, A)
    total_units = N * tensor_a_tiles * tensor_w_tiles * B

    grid = input_tensor.device().compute_with_storage_grid_size()
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, total_units, row_wise=True)

    block_b_rows = min(TILE_ROWS, B, units_per_core_g1)

    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_per_core_g1), (core_group_2, units_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == total_units

    max_blocks_per_core = max(_count_blocks(s, n, B, block_b_rows) for _, s, n in assignment)
    depth_in = min(MAX_DEPTH_IN, max_blocks_per_core)

    core_roles = {f"{c.x},{c.y}": f"rows_tile reader+writer units [{s}, {s + n})" for c, s, n in assignment}

    return Plan(
        regime="rows_tile",
        num_cores=num_cores,
        total_units=total_units,
        units_per_core_g1=units_per_core_g1,
        units_per_core_g2=units_per_core_g2,
        a_tile_rows=a_tile_rows,
        block_b_rows=block_b_rows,
        depth_in=depth_in,
        max_blocks_per_core=max_blocks_per_core,
        core_roles=core_roles,
        shape=(N, A, B, W),
        tensor_a_tiles=tensor_a_tiles,
        tensor_b_tiles=tensor_b_tiles,
        tensor_w_tiles=tensor_w_tiles,
        all_cores=all_cores,
        assignment=assignment,
    )


def create_program_descriptor(plan: Plan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for one plan. Buffer addresses are common runtime args (index 0) so a cached
    descriptor is re-dispatched by patching them; everything else is fixed per plan."""
    N, A, B, W = plan.shape
    tile_bytes = input_tensor.buffer_page_size()
    face_row_bytes = FACE_COLS * input_tensor.element_size()
    assert tile_bytes == output_tensor.buffer_page_size()
    assert tile_bytes == 4 * FACE_COLS * face_row_bytes

    def cb(index, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=input_tensor.dtype, page_size=tile_bytes)
            ],
        )

    cbs = [
        cb(CB_INPUT_ROWS, plan.depth_in * plan.a_tile_rows),
        cb(CB_OUTPUT_STAGING, DEPTH_OUT * plan.block_b_rows),
    ]

    geometry_ct_args = [
        A,
        B,
        plan.tensor_a_tiles,
        plan.tensor_b_tiles,
        plan.tensor_w_tiles,
        plan.a_tile_rows,
        plan.block_b_rows,
        face_row_bytes,
        tile_bytes,
    ]

    reader_ct_args = [CB_INPUT_ROWS, *geometry_ct_args]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [CB_INPUT_ROWS, CB_OUTPUT_STAGING, *geometry_ct_args]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    per_core_rt_args = ttnn.RuntimeArgs()
    for core, start, count in plan.assignment:
        per_core_rt_args[core.x][core.y] = [start, count]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=per_core_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_permute_writer.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=per_core_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)


READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1
