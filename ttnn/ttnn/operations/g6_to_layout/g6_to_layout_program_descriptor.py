# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout program descriptor: regime R1 `tilerow_colgroup_untilize`.

Work unit `(column group j, tile_row)`, unit id `j * tensor_tile_rows + tile_row`
(column-group-major), split over the grid with `split_work_to_cores(row_wise=True)`.
Each core walks its unit range in blocks of `block_height_tiles` tile-rows x
`block_width_tiles` tile columns: reader loads the live faces, compute untilizes,
writer stores the live rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Block knobs. Every CB size, CT arg and RT arg derives from these and from plan().
TILES_IN_DEPTH = 2
ROWS_OUT_DEPTH = 2
L1_OP_BUDGET_BYTES = 1347776
COL_SPLIT_TOLERANCE = 0.10

TILE_DIM = 32
# The CBs carry bf16 bytes declared UInt16 so the unpack -> MOVA2D -> pack path is an opaque
# 16-bit move; a float format would flush subnormals and -0.0 (ckernel_defs.h).
CB_DATA_FORMAT = ttnn.uint16

CB_TILES_IN = 0
CB_ROWS_OUT = 1

REGIME_R1 = "tilerow_colgroup_untilize"


@dataclass(frozen=True)
class Plan:
    regime: str
    grid: tuple  # (x, y)
    shape: tuple
    height: int
    width: int
    image_tile_rows: int  # Ht = ceil(H / 32)
    tensor_tile_rows: int  # N * Ht
    tensor_tile_cols: int  # Wt = ceil(W / 32)
    tile_bytes: int
    block_tiles_cap: int
    num_col_groups: int
    col_group_width_tiles: int
    block_height_tiles: int
    block_width_tiles: int
    tiles_in_depth: int
    rows_out_depth: int
    num_units: int
    num_cores: int
    units_per_core: tuple  # (group 1, group 2)
    core_ranges: tuple  # ((x0, y0), (x1, y1)) rectangles of the used cores
    core_assignment: tuple  # ((x, y), unit_begin, unit_end, num_blocks) per used core

    @property
    def block_tiles(self) -> int:
        return self.block_height_tiles * self.block_width_tiles

    def report(self) -> dict:
        used = {xy for xy, *_ in self.core_assignment}
        grid_x, grid_y = self.grid
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "num_col_groups": self.num_col_groups,
            "col_group_width_tiles": self.col_group_width_tiles,
            "block_height_tiles": self.block_height_tiles,
            "block_width_tiles": self.block_width_tiles,
            "units_per_core": self.units_per_core,
            "core_roles": {
                (x, y): ("reader/compute/writer" if (x, y) in used else "idle")
                for y in range(grid_y)
                for x in range(grid_x)
            },
        }


def choose_column_groups(tensor_tile_rows, tensor_tile_cols, num_grid_cores, tolerance):
    """Smallest number of column groups whose most-loaded core is within `tolerance` of balance."""
    balanced_load = math.ceil(tensor_tile_rows * tensor_tile_cols / num_grid_cores)
    for groups in range(1, tensor_tile_cols + 1):
        group_width = math.ceil(tensor_tile_cols / groups)
        effective_groups = math.ceil(tensor_tile_cols / group_width)
        cost = math.ceil(tensor_tile_rows * effective_groups / num_grid_cores) * group_width
        if cost <= (1 + tolerance) * balanced_load:
            return effective_groups, group_width
    return tensor_tile_cols, 1


def count_blocks(unit_begin, unit_end, plan_fields):
    """Host twin of the kernels' block walk; must match it exactly."""
    tensor_tile_rows, tensor_tile_cols, group_width, block_h, block_w = plan_fields
    blocks = 0
    unit = unit_begin
    while unit < unit_end:
        group = unit // tensor_tile_rows
        tile_row = unit - group * tensor_tile_rows
        rows = min(block_h, tensor_tile_rows - tile_row, unit_end - unit)
        col_begin = group * group_width
        col_end = min(col_begin + group_width, tensor_tile_cols)
        blocks += math.ceil((col_end - col_begin) / block_w)
        unit += rows
    return blocks


def compute_plan(
    shape,
    grid,
    *,
    output_in_l1,
    num_l1_banks,
    l1_alignment,
    tiles_in_depth=TILES_IN_DEPTH,
    rows_out_depth=ROWS_OUT_DEPTH,
    col_split_tolerance=COL_SPLIT_TOLERANCE,
    l1_op_budget_bytes=L1_OP_BUDGET_BYTES,
) -> Plan:
    """Pure function of shape, output placement and grid (never of measured free L1)."""
    shape = tuple(int(d) for d in shape)
    height, width = shape[-2], shape[-1]
    images = math.prod(shape[:-2])
    image_tile_rows = math.ceil(height / TILE_DIM)
    tensor_tile_rows = images * image_tile_rows
    tensor_tile_cols = math.ceil(width / TILE_DIM)
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    out_l1_bytes_per_bank = 0
    if output_in_l1:
        row_stride = math.ceil(width * 2 / l1_alignment) * l1_alignment
        out_l1_bytes_per_bank = math.ceil(images * height / num_l1_banks) * row_stride
    block_tiles_cap = (l1_op_budget_bytes - out_l1_bytes_per_bank) // ((tiles_in_depth + rows_out_depth) * tile_bytes)
    if block_tiles_cap < 1:
        raise RuntimeError(
            f"g6_to_layout: the L1 output takes {out_l1_bytes_per_bank} B per bank, leaving no room for one "
            f"block within the {l1_op_budget_bytes} B the model leaves to the op"
        )

    grid_x, grid_y = grid
    num_grid_cores = grid_x * grid_y
    num_col_groups, col_group_width_tiles = choose_column_groups(
        tensor_tile_rows, tensor_tile_cols, num_grid_cores, col_split_tolerance
    )
    num_units = tensor_tile_rows * num_col_groups

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_1,
        units_per_core_2,
    ) = ttnn.split_work_to_cores(ttnn.CoreCoord(grid_x, grid_y), num_units, row_wise=True)

    core_rows_max = min(math.ceil(num_units / num_cores), tensor_tile_rows)
    block_width_tiles = math.ceil(col_group_width_tiles / math.ceil(col_group_width_tiles / block_tiles_cap))
    cap_rows = block_tiles_cap // block_width_tiles
    block_height_tiles = math.ceil(core_rows_max / math.ceil(core_rows_max / cap_rows))

    walk_fields = (tensor_tile_rows, tensor_tile_cols, col_group_width_tiles, block_height_tiles, block_width_tiles)
    assignment = []
    unit_begin = 0
    for group, units in ((core_group_1, units_per_core_1), (core_group_2, units_per_core_2)):
        if units == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            unit_end = unit_begin + units
            assignment.append(((core.x, core.y), unit_begin, unit_end, count_blocks(unit_begin, unit_end, walk_fields)))
            unit_begin = unit_end
    assert unit_begin == num_units, f"g6_to_layout: split covers {unit_begin} of {num_units} units"

    return Plan(
        regime=REGIME_R1,
        grid=(grid_x, grid_y),
        shape=shape,
        height=height,
        width=width,
        image_tile_rows=image_tile_rows,
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        tile_bytes=tile_bytes,
        block_tiles_cap=block_tiles_cap,
        num_col_groups=num_col_groups,
        col_group_width_tiles=col_group_width_tiles,
        block_height_tiles=block_height_tiles,
        block_width_tiles=block_width_tiles,
        tiles_in_depth=tiles_in_depth,
        rows_out_depth=rows_out_depth,
        num_units=num_units,
        num_cores=num_cores,
        units_per_core=(units_per_core_1, units_per_core_2),
        core_ranges=tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in all_cores.ranges()),
        core_assignment=tuple(assignment),
    )


def _core_range_set(plan: Plan) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end)) for start, end in plan.core_ranges]
    )


def _cb(index, num_pages, plan: Plan, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * plan.tile_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=CB_DATA_FORMAT, page_size=plan.tile_bytes)
        ],
    )


class CachedProgram:
    """A built descriptor whose only per-call state is the two buffer addresses (common RT args)."""

    def __init__(self, reader, writer, compute, cbs):
        self._reader = reader
        self._writer = writer
        self._compute = compute
        self._cbs = cbs

    def descriptor_for(self, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
        self._reader.common_runtime_args = [input_tensor.buffer_address()]
        self._writer.common_runtime_args = [output_tensor.buffer_address()]
        return ttnn.ProgramDescriptor(kernels=[self._reader, self._writer, self._compute], semaphores=[], cbs=self._cbs)


def create_program(plan: Plan, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> CachedProgram:
    cores = _core_range_set(plan)

    cbs = [
        _cb(CB_TILES_IN, plan.tiles_in_depth * plan.block_tiles, plan, cores),
        _cb(CB_ROWS_OUT, plan.rows_out_depth * plan.block_tiles, plan, cores),
    ]

    walk_ct_args = [
        plan.block_height_tiles,
        plan.block_width_tiles,
        plan.tensor_tile_rows,
        plan.tensor_tile_cols,
        plan.image_tile_rows,
        plan.height,
        plan.width,
        plan.col_group_width_tiles,
        plan.tile_bytes,
    ]
    reader_ct_args = [CB_TILES_IN, *walk_ct_args]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [CB_ROWS_OUT, *walk_ct_args]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct_args = [CB_TILES_IN, CB_ROWS_OUT, plan.block_width_tiles]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for (x, y), unit_begin, unit_end, num_blocks in plan.core_assignment:
        reader_rt_args[x][y] = [unit_begin, unit_end]
        writer_rt_args[x][y] = [unit_begin, unit_end]
        compute_rt_args[x][y] = [num_blocks * plan.block_height_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(),
    )
    return CachedProgram(reader, writer, compute, cbs)
