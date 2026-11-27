# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute, regime rows_tile_scatter: plan (input-tile split, block knobs) and ProgramDescriptor.

Work unit: one input tile, the page p_in(n, a, bt, wt) = ((n * tensor_a + a) * tensor_bt + bt) * tensor_wt + wt.
Each core owns a contiguous page range and cuts it into blocks of block_in_tiles pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME = "rows_tile_scatter"

TILE_ROWS = 32
FACE_COLS = 16

CB_INPUT_TILES = 0

# Buffer-depth knob of cb_input_tiles; the allocated depth is min(INPUT_DEPTH, max_blocks_per_core).
INPUT_DEPTH = 2
# L1 the model leaves per core, the output's own L1 pages included (BB-05).
L1_OP_BUDGET_BYTES = 1347776
# Tuning cap on block_in_tiles on top of the L1 cap; None keeps the design value
# min(max_core_in_tiles, block_cap).
BLOCK_IN_TILES_LIMIT = None
# Each tile's row walk starts at a mod rows_live(bt), so cores with neighbouring a write different pages first.
STAGGER_ROWS = 1

READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1


@dataclass
class Plan:
    regime: str
    num_cores: int
    tensor_in_tiles: int
    tiles_per_core_g1: int
    tiles_per_core_g2: int
    block_in_tiles: int
    block_cap: int
    input_depth: int
    max_blocks_per_core: int
    num_blocks_g1: int
    num_blocks_g2: int
    stagger_rows: int
    core_roles: dict
    shape: tuple = ()
    tensor_at: int = 0
    tensor_bt: int = 0
    tensor_wt: int = 0
    all_cores: object = None
    assignment: list = field(default_factory=list)  # [(CoreCoord, start_page, num_pages)]


def _div_up(a, b):
    return (a + b - 1) // b


def worker_grid(device):
    """The grid the input-tile split spreads over; part of the program cache key."""
    return device.compute_with_storage_grid_size()


def out_l1_bytes_per_core(output_buffer_type, out_num_pages, out_bank_page_bytes, device):
    """The output's own L1 pages on one core; they count against L1_OP_BUDGET_BYTES (BB-05)."""
    if output_buffer_type != ttnn.BufferType.L1:
        return 0
    num_l1_banks = ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks
    return _div_up(out_num_pages, num_l1_banks) * out_bank_page_bytes


def split_units(grid, num_units):
    """split_work_to_cores over the full grid (row_wise=True) plus each core's contiguous unit range,
    cores walked row-major: (num_cores, all_cores, units_g1, units_g2, [(core, start, count)])."""
    num_cores, all_cores, core_group_1, core_group_2, units_g1, units_g2 = ttnn.split_work_to_cores(
        grid, num_units, row_wise=True
    )
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_g1), (core_group_2, units_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == num_units
    return num_cores, all_cores, units_g1, units_g2, assignment


def knob_key():
    """Host knobs that reach this regime's CT args; part of the program cache key."""
    return (INPUT_DEPTH, BLOCK_IN_TILES_LIMIT, STAGGER_ROWS)


def build_plan(input_tensor, canonical_dims) -> Plan:
    assert tuple(canonical_dims) == (0, 2, 1, 3), f"only regime {REGIME} is built"
    assert INPUT_DEPTH >= 1
    N, A, B, W = (int(d) for d in input_tensor.shape)
    tensor_at = _div_up(A, TILE_ROWS)
    tensor_bt = _div_up(B, TILE_ROWS)
    tensor_wt = _div_up(W, TILE_ROWS)
    tensor_in_tiles = N * A * tensor_bt * tensor_wt
    tensor_out_tiles = N * B * tensor_at * tensor_wt
    assert tensor_in_tiles == input_tensor.buffer_num_pages()
    in_page_bytes = input_tensor.buffer_page_size()

    out_l1 = out_l1_bytes_per_core(
        input_tensor.memory_config().buffer_type, tensor_out_tiles, in_page_bytes, input_tensor.device()
    )
    block_cap = (L1_OP_BUDGET_BYTES - out_l1) // (INPUT_DEPTH * in_page_bytes)
    if block_cap < 1:
        raise RuntimeError(
            f"g6_permute: shape {[N, A, B, W]} leaves no room for one input block within "
            f"{L1_OP_BUDGET_BYTES} B of L1 per core (output pages take {out_l1} B)"
        )

    num_cores, all_cores, tiles_g1, tiles_g2, assignment = split_units(
        worker_grid(input_tensor.device()), tensor_in_tiles
    )

    block_in_tiles = min(tiles_g1, block_cap)
    if BLOCK_IN_TILES_LIMIT is not None:
        block_in_tiles = min(block_in_tiles, BLOCK_IN_TILES_LIMIT)
    assert block_in_tiles >= 1
    max_blocks_per_core = _div_up(tiles_g1, block_in_tiles)
    input_depth = min(INPUT_DEPTH, max_blocks_per_core)

    core_roles = {f"{c.x},{c.y}": f"{REGIME} reader+writer input pages [{s}, {s + n})" for c, s, n in assignment}

    return Plan(
        regime=REGIME,
        num_cores=num_cores,
        tensor_in_tiles=tensor_in_tiles,
        tiles_per_core_g1=tiles_g1,
        tiles_per_core_g2=tiles_g2,
        block_in_tiles=block_in_tiles,
        block_cap=block_cap,
        input_depth=input_depth,
        max_blocks_per_core=max_blocks_per_core,
        num_blocks_g1=_div_up(tiles_g1, block_in_tiles),
        num_blocks_g2=_div_up(tiles_g2, block_in_tiles),
        stagger_rows=STAGGER_ROWS,
        core_roles=core_roles,
        shape=(N, A, B, W),
        tensor_at=tensor_at,
        tensor_bt=tensor_bt,
        tensor_wt=tensor_wt,
        all_cores=all_cores,
        assignment=assignment,
    )


def _start_coords(plan: Plan, page):
    """(n, a, bt, wt) of input page `page`; the kernels advance them without dividing."""
    wt = page % plan.tensor_wt
    rest = page // plan.tensor_wt
    bt = rest % plan.tensor_bt
    rest //= plan.tensor_bt
    A = plan.shape[1]
    return rest // A, rest % A, bt, wt


def create_program_descriptor(plan: Plan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    """Descriptor for one plan. Buffer addresses are common runtime args (index 0) so a cached
    descriptor is re-dispatched by patching them; everything else is fixed per plan."""
    N, A, B, W = plan.shape
    in_page_bytes = input_tensor.buffer_page_size()
    face_row_bytes = FACE_COLS * input_tensor.element_size()
    assert in_page_bytes == output_tensor.buffer_page_size()
    assert in_page_bytes == 4 * FACE_COLS * face_row_bytes

    cbs = [
        ttnn.CBDescriptor(
            total_size=plan.input_depth * plan.block_in_tiles * in_page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_page_bytes
                )
            ],
        )
    ]

    shared_ct_args = [
        CB_INPUT_TILES,
        A,
        B,
        plan.tensor_at,
        plan.tensor_bt,
        plan.tensor_wt,
        plan.block_in_tiles,
        face_row_bytes,
        in_page_bytes,
        plan.stagger_rows,
    ]
    reader_ct_args = [*shared_ct_args, *ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()]
    writer_ct_args = [*shared_ct_args, *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()]

    per_core_rt_args = ttnn.RuntimeArgs()
    for core, start, count in plan.assignment:
        per_core_rt_args[core.x][core.y] = [start, count, *_start_coords(plan, start)]

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
    kernels = [None, None]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
