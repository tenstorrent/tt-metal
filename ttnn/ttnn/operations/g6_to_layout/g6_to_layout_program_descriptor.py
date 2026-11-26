# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plan and ProgramDescriptor for g6_to_layout, regime R1 `untilize_interleaved`.

Work unit: a segment = one tile-row x `segment_width` consecutive tile-columns. Segments are
spread over the full compute grid in contiguous ranges; each core streams its range through
reader -> cb_tiles -> untilize -> cb_rows -> writer in blocks of `block_segments` segments.
Both CBs are declared UInt16 so MOVA2D moves the bf16 bits without flushing zero-exponent
datums (subnormals, -0.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# The model leaves this much L1 per core to the op (an L1 output's own pages count inside it).
L1_BUDGET_BYTES = 1347776
# One 64-B base alignment per CB: Blackhole DRAM reads need local == NoC address mod 64.
CB_BASE_ALIGN_BYTES = 64

TILES_DEPTH = 2
ROWS_DEPTH = 2
BALANCE_SLACK = Fraction(1, 8)

CB_TILES = 0  # input tile pages (bit view of bf16), reader -> compute
CB_ROWS = 1  # untilized rows of one segment, compute -> writer
NUM_CBS = 2

# Kernel order inside the descriptor; the entry point patches the buffer address of these two.
READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1
COMPUTE_KERNEL_INDEX = 2

REGIME_UNTILIZE_INTERLEAVED = "untilize_interleaved"


@dataclass(frozen=True)
class Plan:
    regime: str
    grid: tuple  # (x, y) of the compute grid
    # Geometry
    shape: tuple
    tensor_n: int
    height: int
    width: int
    tensor_h_tiles: int
    tensor_w_tiles: int
    tensor_tile_rows: int
    # Block knobs
    num_col_chunks: int
    segment_width: int
    num_segments: int
    block_segments: int
    tiles_depth: int
    rows_depth: int
    balance_slack: Fraction
    # Formats / budget
    input_dtype: object
    output_dtype: object
    output_memory_config: object
    tile_bytes_in: int
    tile_bytes_rows: int
    out_l1_bank_bytes: int
    cb_budget_bytes: int
    # Work split
    cores_used: int
    group_1: tuple  # ((x, y), ...) in assignment order
    group_1_segments: int
    group_2: tuple
    group_2_segments: int
    core_assignment: tuple  # ((x, y), start_segment, num_segments) per used core
    core_roles: tuple  # ((x, y), "reader/compute/writer" | "idle") over the whole grid

    @property
    def cb_tiles_pages(self) -> int:
        return self.tiles_depth * self.block_segments * self.segment_width

    @property
    def cb_rows_pages(self) -> int:
        return self.rows_depth * self.block_segments * self.segment_width

    @property
    def l1_footprint_bytes(self) -> int:
        return (
            self.cb_tiles_pages * self.tile_bytes_in
            + self.cb_rows_pages * self.tile_bytes_rows
            + self.out_l1_bank_bytes
            + NUM_CBS * CB_BASE_ALIGN_BYTES
        )


def _choose_segment_width(tensor_tile_rows, tensor_w_tiles, grid_cores, seg_width_max, balance_slack):
    """Smallest column-chunk count whose per-core load is within balance_slack of ideal."""
    ideal = math.ceil(tensor_tile_rows * tensor_w_tiles / grid_cores)
    limit = ideal * (1 + balance_slack)
    for col_chunks in range(1, tensor_w_tiles + 1):
        width = math.ceil(tensor_w_tiles / col_chunks)
        if width > seg_width_max:
            continue
        chunks = math.ceil(tensor_w_tiles / width)
        num_segments = tensor_tile_rows * chunks
        if math.ceil(num_segments / grid_cores) * width <= limit:
            return width, chunks, num_segments
    raise AssertionError("unreachable: one tile per segment always balances")


def compute_plan(
    shape,
    input_dtype,
    output_dtype,
    output_memory_config,
    grid_x,
    grid_y,
    num_l1_banks,
    l1_alignment,
) -> Plan:
    """Pure function of shape, dtypes, placements and grid (never of measured free L1)."""
    shape = tuple(int(d) for d in shape)
    height, width = shape[-2], shape[-1]
    tensor_n = math.prod(shape[:-2]) if len(shape) > 2 else 1
    tensor_h_tiles = math.ceil(height / TILE_DIM)
    tensor_w_tiles = math.ceil(width / TILE_DIM)
    tensor_tile_rows = tensor_n * tensor_h_tiles

    tile_bytes_in = ttnn.tile_size(input_dtype)
    tile_bytes_rows = ttnn.tile_size(output_dtype)

    if output_memory_config.buffer_type == ttnn.BufferType.L1:
        out_row_stride = math.ceil(width * 2 / l1_alignment) * l1_alignment
        out_l1_bank_bytes = math.ceil(tensor_n * height / num_l1_banks) * out_row_stride
    else:
        out_l1_bank_bytes = 0
    cb_budget = L1_BUDGET_BYTES - out_l1_bank_bytes - NUM_CBS * CB_BASE_ALIGN_BYTES
    bytes_per_tile_column = TILES_DEPTH * tile_bytes_in + ROWS_DEPTH * tile_bytes_rows
    if cb_budget < bytes_per_tile_column:
        raise RuntimeError(
            f"g6_to_layout: output occupies {out_l1_bank_bytes} B of L1 per bank, leaving {cb_budget} B "
            f"for buffers; one segment of one tile needs {bytes_per_tile_column} B"
        )
    seg_width_max = cb_budget // bytes_per_tile_column

    grid_cores = grid_x * grid_y
    segment_width, num_col_chunks, num_segments = _choose_segment_width(
        tensor_tile_rows, tensor_w_tiles, grid_cores, seg_width_max, BALANCE_SLACK
    )
    max_segments_per_core = math.ceil(num_segments / min(grid_cores, num_segments))
    block_segments = max(1, min(max_segments_per_core, cb_budget // (segment_width * bytes_per_tile_column)))

    (
        cores_used,
        _all_cores,
        core_group_1,
        core_group_2,
        group_1_segments,
        group_2_segments,
    ) = ttnn.split_work_to_cores(ttnn.CoreCoord(grid_x, grid_y), num_segments, row_wise=True)
    group_1 = tuple((c.x, c.y) for c in ttnn.corerange_to_cores(core_group_1, row_wise=True))
    group_2 = tuple((c.x, c.y) for c in ttnn.corerange_to_cores(core_group_2, row_wise=True))

    assignment = []
    start = 0
    for cores, count in ((group_1, group_1_segments), (group_2, group_2_segments)):
        for core in cores:
            assignment.append((core, start, count))
            start += count
    assert start == num_segments, f"g6_to_layout: split covers {start} of {num_segments} segments"

    used = {core for core, _, _ in assignment}
    roles = tuple(
        ((x, y), "reader/compute/writer" if (x, y) in used else "idle") for y in range(grid_y) for x in range(grid_x)
    )

    return Plan(
        regime=REGIME_UNTILIZE_INTERLEAVED,
        grid=(grid_x, grid_y),
        shape=shape,
        tensor_n=tensor_n,
        height=height,
        width=width,
        tensor_h_tiles=tensor_h_tiles,
        tensor_w_tiles=tensor_w_tiles,
        tensor_tile_rows=tensor_tile_rows,
        num_col_chunks=num_col_chunks,
        segment_width=segment_width,
        num_segments=num_segments,
        block_segments=block_segments,
        tiles_depth=TILES_DEPTH,
        rows_depth=ROWS_DEPTH,
        balance_slack=BALANCE_SLACK,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        output_memory_config=output_memory_config,
        tile_bytes_in=tile_bytes_in,
        tile_bytes_rows=tile_bytes_rows,
        out_l1_bank_bytes=out_l1_bank_bytes,
        cb_budget_bytes=cb_budget,
        cores_used=cores_used,
        group_1=group_1,
        group_1_segments=group_1_segments,
        group_2=group_2,
        group_2_segments=group_2_segments,
        core_assignment=tuple(assignment),
        core_roles=roles,
    )


def _core_range_set(plan: Plan) -> ttnn.CoreRangeSet:
    _, all_cores, _, _, _, _ = ttnn.split_work_to_cores(ttnn.CoreCoord(*plan.grid), plan.num_segments, row_wise=True)
    return all_cores


def create_program_descriptor(plan: Plan, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor):
    """Descriptor for `plan`; buffer addresses live in each data-movement kernel's common runtime args."""
    all_cores = _core_range_set(plan)

    # Both CBs are UInt16: the bf16 bits are moved, never interpreted (see module docstring).
    cb_tiles = ttnn.CBDescriptor(
        total_size=plan.cb_tiles_pages * plan.tile_bytes_in,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_TILES, data_format=ttnn.uint16, page_size=plan.tile_bytes_in)
        ],
    )
    cb_rows = ttnn.CBDescriptor(
        total_size=plan.cb_rows_pages * plan.tile_bytes_rows,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ROWS, data_format=ttnn.uint16, page_size=plan.tile_bytes_rows)
        ],
    )

    reader_ct_args = [
        CB_TILES,
        plan.segment_width,
        plan.block_segments,
        plan.num_col_chunks,
        plan.tensor_h_tiles,
        plan.tensor_w_tiles,
        plan.height,
        plan.width,
        plan.tile_bytes_in,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_ROWS,
        plan.segment_width,
        plan.block_segments,
        plan.num_col_chunks,
        plan.tensor_h_tiles,
        plan.height,
        plan.width,
        plan.tile_bytes_rows,
        output_tensor.buffer_page_size(),
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [CB_TILES, CB_ROWS, plan.segment_width]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for (x, y), start, count in plan.core_assignment:
        reader_rt[x][y] = [start, count]
        writer_rt[x][y] = [start, count]
        compute_rt[x][y] = [count]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = False
    compute_config.dst_full_sync_en = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_to_layout_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_config,
    )

    kernels = [None] * 3
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    kernels[COMPUTE_KERNEL_INDEX] = compute
    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_tiles, cb_rows])
    # Buffer addresses are runtime args, so the structural hash is fixed per plan; memoize it.
    descriptor.custom_program_hash = ttnn.compute_program_descriptor_hash(descriptor)
    return descriptor


def patch_buffer_addresses(descriptor, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> None:
    descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args = [input_tensor.buffer_address()]
    descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args = [output_tensor.buffer_address()]
