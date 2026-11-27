# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_unsqueeze program descriptor: the plan and the one-kernel program built from it.

Regime `row_gather_face_scatter` (op_design.md, Blocking Model): one data-movement kernel per
core (the mover, on BRISC) reads face row 0 of faces 0 and 1 of each of its input tiles into the
bases of output faces 0 and 2 of a staging page, moves the lanes into column 0 with 32-bit word
copies in place, and writes the whole staging page as the output tile. There is no compute kernel.

`_plan()` is the single source of every derived quantity; `create_program_descriptor()` and
`plan_for()` (op file) both consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
MOVER_KERNEL = KERNEL_DIR / "g6_unsqueeze_mover.cpp"

REGIME = "row_gather_face_scatter"

# Per-core L1 the model leaves to the op (contract BB-05); bounds the staging block.
L1_BUDGET_BYTES = 1347776

# Core-count knob. None spreads the tiles over the whole compute grid.
MAX_CORES = None

# The op's only CB: staging pages that become output tiles in place.
CB_OUTPUT_TILES = 0


@dataclass(frozen=True)
class CoreWork:
    x: int
    y: int
    start_tile: int
    core_w_tiles: int

    @property
    def role(self) -> str:
        return "mover" if self.core_w_tiles > 0 else "idle_tail"


@dataclass(frozen=True)
class Plan:
    regime: str
    grid: tuple
    max_cores: int
    rect_cols: int
    rect_rows: int
    tensor_w_tiles: int
    cores: tuple  # CoreWork, row-major over the rectangle
    block_w_tiles: int
    staging_depth_blocks: int
    tile_bytes: int
    face_bytes: int
    face_row_bytes: int

    @property
    def core_range_set(self) -> ttnn.CoreRangeSet:
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.rect_cols - 1, self.rect_rows - 1))]
        )

    @property
    def l1_bytes_per_core(self) -> int:
        return self.staging_depth_blocks * self.block_w_tiles * self.tile_bytes


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def _block_shape(core_w_tiles_max: int, tile_bytes: int) -> tuple:
    """(block_w_tiles, staging_depth_blocks): the whole per-core assignment as one block while it
    fits the budget at depth 1; otherwise balanced blocks double-buffered within the budget."""
    max_single_block_tiles = L1_BUDGET_BYTES // tile_bytes
    if core_w_tiles_max <= max_single_block_tiles:
        return core_w_tiles_max, 1
    max_double_block_tiles = L1_BUDGET_BYTES // (2 * tile_bytes)
    num_blocks = _div_up(core_w_tiles_max, max_double_block_tiles)
    return _div_up(core_w_tiles_max, num_blocks), 2


def _plan(input_tensor: ttnn.Tensor, max_cores=None) -> Plan:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    if max_cores is None:
        max_cores = MAX_CORES if MAX_CORES is not None else grid_cores
    if not 1 <= max_cores <= grid_cores:
        raise RuntimeError(f"g6_unsqueeze: max_cores={max_cores} outside 1..{grid_cores}")

    element_bytes = input_tensor.element_size()
    if element_bytes != 4:
        # One lane is one 32-bit word, and a 64 B face row meets the DRAM read alignment.
        raise RuntimeError(f"g6_unsqueeze: the move needs 4-byte elements, got {element_bytes}")
    tile_h, tile_w = input_tensor.tile.tile_shape
    face_h, face_w = input_tensor.tile.face_shape
    tile_bytes = tile_h * tile_w * element_bytes
    face_bytes = face_h * face_w * element_bytes
    face_row_bytes = face_w * element_bytes
    if input_tensor.buffer_page_size() != tile_bytes:
        raise RuntimeError(
            f"g6_unsqueeze: input page is {input_tensor.buffer_page_size()} B, expected one {tile_bytes} B tile"
        )
    if face_row_bytes % ttnn.get_dram_alignment() != 0:
        raise RuntimeError(
            f"g6_unsqueeze: a {face_row_bytes} B face-row read breaks the {ttnn.get_dram_alignment()} B DRAM alignment"
        )

    tensor_w_tiles = _div_up(input_tensor.shape[-1], tile_w)
    if input_tensor.buffer_num_pages() != tensor_w_tiles:
        raise RuntimeError(
            f"g6_unsqueeze: input holds {input_tensor.buffer_num_pages()} pages, expected {tensor_w_tiles}"
        )

    # A wide rectangle with origin (0, 0): one core range per call.
    num_work_cores = min(tensor_w_tiles, max_cores)
    rect_rows = _div_up(num_work_cores, grid.x)
    rect_cols = _div_up(num_work_cores, rect_rows)

    num_cores, _, core_group_1, core_group_2, units_g1, units_g2 = ttnn.split_work_to_cores(
        ttnn.CoreCoord(rect_cols, rect_rows), tensor_w_tiles, row_wise=True
    )
    cores = []
    start_tile = 0
    for y in range(rect_rows):
        for x in range(rect_cols):
            coord = ttnn.CoreCoord(x, y)
            if core_group_1.contains(coord):
                units = units_g1
            elif core_group_2.contains(coord):
                units = units_g2
            else:
                units = 0
            cores.append(CoreWork(x, y, start_tile, units))
            start_tile += units
    if start_tile != tensor_w_tiles or sum(1 for c in cores if c.core_w_tiles) != num_cores:
        raise RuntimeError(f"g6_unsqueeze: work split covers {start_tile} of {tensor_w_tiles} tiles")

    block_w_tiles, staging_depth_blocks = _block_shape(max(units_g1, units_g2), tile_bytes)
    return Plan(
        regime=REGIME,
        grid=(grid.x, grid.y),
        max_cores=max_cores,
        rect_cols=rect_cols,
        rect_rows=rect_rows,
        tensor_w_tiles=tensor_w_tiles,
        cores=tuple(cores),
        block_w_tiles=block_w_tiles,
        staging_depth_blocks=staging_depth_blocks,
        tile_bytes=tile_bytes,
        face_bytes=face_bytes,
        face_row_bytes=face_row_bytes,
    )


def create_program_descriptor(plan: Plan, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> tuple:
    """Build the program for `plan`. Returns (program_descriptor, mover_kernel).

    The buffer addresses are the kernel's common runtime args and appear nowhere else, so the
    caller can reuse both objects for any pair of tensors with the same plan: set
    `mover_kernel.common_runtime_args` and reassign `program_descriptor.kernels`.
    """
    if output_tensor.buffer_num_pages() != plan.tensor_w_tiles or output_tensor.buffer_page_size() != plan.tile_bytes:
        raise RuntimeError(
            f"g6_unsqueeze: output holds {output_tensor.buffer_num_pages()} pages of "
            f"{output_tensor.buffer_page_size()} B, expected {plan.tensor_w_tiles} of {plan.tile_bytes} B"
        )
    core_ranges = plan.core_range_set

    cb_output_tiles = ttnn.CBDescriptor(
        total_size=plan.l1_bytes_per_core,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=plan.tile_bytes,
            )
        ],
    )

    ct_args = [
        CB_OUTPUT_TILES,
        plan.block_w_tiles,
        plan.staging_depth_blocks,
        plan.tile_bytes,
        plan.face_bytes,
        plan.face_row_bytes,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for core in plan.cores:
        rt_args[core.x][core.y] = [core.start_tile, core.core_w_tiles]

    mover_kernel = ttnn.KernelDescriptor(
        kernel_source=str(MOVER_KERNEL),
        core_ranges=core_ranges,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[mover_kernel],
        semaphores=[],
        cbs=[cb_output_tiles],
    )
    return program_descriptor, mover_kernel
