# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm_tile ProgramDescriptor: regime row_resident, tile-rows split across the compute grid."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# Blocking-model host constants (op_design.md, Blocking Model). Each is defined once.
CB_L1_BUDGET_BYTES = 768 * 1024
BLOCK_HEIGHT_MAX = 1
INPUT_DEPTH_MAX = 4
# Tile-row count from which the residual stream moves to the writer RISC / other NoC. Measured on BH
# p100a (residual, us, split off -> on): R=32 17.4-18.3 -> 19.2-20.6, R=64 33 -> 26.5, R=128 58 -> 36.4.
RESIDUAL_SPLIT_MIN_TILE_ROWS = 64
OUTPUT_DEPTH = 2

CB_INPUT_TILES = 0
CB_RESIDUAL_TILES = 1
CB_GAMMA_ROWS = 2
CB_BETA_ROWS = 3
# Alias group `work`: one allocation, lockstep ids (op_design.md, Circular Buffers).
CB_XSUM = 4
CB_CENTERED = 5
CB_NORMED = 6
CB_SCALED = 7
# Alias group `stats`.
CB_MEAN = 8
CB_SUMSQ = 9
CB_RSTD = 10
CB_OUTPUT_TILES = 16

WORK_GROUP = (CB_XSUM, CB_CENTERED, CB_NORMED, CB_SCALED)
STATS_GROUP = (CB_MEAN, CB_SUMSQ, CB_RSTD)


@dataclass(frozen=True)
class Blocking:
    block_height: int
    input_depth: int
    tensor_w_tiles: int
    in_tile_bytes: int
    acc_dtype: object
    has_residual: bool
    # Residual stream fetched by the writer RISC on the other NoC (else by the reader, beside x).
    residual_on_writer: bool
    # {allocation: (pages, page_bytes)} from _cb_plan
    cb_plan: dict
    all_cores: object
    # ((core, start_row, num_blocks, last_block_extent), ...)
    assignment: tuple


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _split_rows(device, tensor_rows_tiles):
    grid = device.compute_with_storage_grid_size()
    _, all_cores, group_1, group_2, rows_g1, rows_g2 = ttnn.split_work_to_cores(grid, tensor_rows_tiles, row_wise=True)
    cores_rows = []
    for group, per_core in ((group_1, rows_g1), (group_2, rows_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            cores_rows.append((core, per_core))
    return all_cores, cores_rows, rows_g1


def _cb_plan(block_height, input_depth, *, Wt, in_bytes, acc_bytes, has_residual):
    """{allocation: (pages, page_bytes)} per core (l1_ledger.md). Single source for CB sizes and the L1 cap."""
    block_tiles = block_height * Wt
    plan = {
        "input": (input_depth * block_tiles, in_bytes),
        "gamma": (Wt, in_bytes),
        "beta": (Wt, in_bytes),
        "work": (block_tiles, acc_bytes),
        "stats": (block_height, acc_bytes),
        "output": (OUTPUT_DEPTH * Wt, in_bytes),
    }
    if has_residual:
        plan["residual"] = (input_depth * block_tiles, in_bytes)
    return plan


def _footprint(block_height, input_depth, **geom):
    return sum(pages * page_bytes for pages, page_bytes in _cb_plan(block_height, input_depth, **geom).values())


def _block_height_cap(input_depth, **geom):
    fixed = _footprint(0, input_depth, **geom)
    per_row = _footprint(1, input_depth, **geom) - fixed
    return (CB_L1_BUDGET_BYTES - fixed) // per_row


def derive_blocking(input_tensor, *, has_residual, fp32_dest_acc_en):
    """Single source of every block knob. Returns None when row_resident's predicate fails."""
    padded = list(input_tensor.padded_shape)
    Wt = padded[-1] // TILE
    tensor_rows_tiles = padded[0] * padded[1] * (padded[2] // TILE)

    in_bytes = input_tensor.tile.get_tile_size(input_tensor.dtype)
    acc_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    acc_bytes = ttnn.tile_size(acc_dtype)
    geom = dict(Wt=Wt, in_bytes=in_bytes, acc_bytes=acc_bytes, has_residual=has_residual)

    if _block_height_cap(1, **geom) < 1:
        return None

    all_cores, cores_rows, max_core_rows_tiles = _split_rows(input_tensor.device(), tensor_rows_tiles)
    # Compute runs BLOCK_HEIGHT_MAX tile-rows per block so it starts as soon as the first rows land; the
    # reader runs up to input_depth blocks ahead (whole assignment when it fits) so reads never wait on it.
    block_height = min(max_core_rows_tiles, BLOCK_HEIGHT_MAX, _block_height_cap(1, **geom))
    input_depth = min(-(-max_core_rows_tiles // block_height), INPUT_DEPTH_MAX)
    while input_depth > 1 and _block_height_cap(input_depth, **geom) < block_height:
        input_depth -= 1

    assignment = []
    start_row = 0
    for core, core_rows_tiles in cores_rows:
        num_blocks = -(-core_rows_tiles // block_height)
        last_block_extent = core_rows_tiles - (num_blocks - 1) * block_height
        assignment.append((core, start_row, num_blocks, last_block_extent))
        start_row += core_rows_tiles

    return Blocking(
        block_height=block_height,
        input_depth=input_depth,
        tensor_w_tiles=Wt,
        in_tile_bytes=in_bytes,
        acc_dtype=acc_dtype,
        has_residual=has_residual,
        residual_on_writer=has_residual and tensor_rows_tiles >= RESIDUAL_SPLIT_MIN_TILE_ROWS,
        cb_plan=_cb_plan(block_height, input_depth, **geom),
        all_cores=all_cores,
        assignment=tuple(assignment),
    )


def _cb(indices, num_pages, page_size, data_format, cores):
    """One allocation; several indices make an alias group (lockstep ids over the same bytes)."""
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)
            for index in indices
        ],
    )


def _affine_ct_args(affine):
    """(is_row_major, stick_elems, page_bytes): page geometry the writer needs to locate 32-element chunks."""
    is_rm = affine.layout == ttnn.ROW_MAJOR_LAYOUT
    stick_elems = list(affine.padded_shape)[-1] if is_rm else 0
    return [int(is_rm), stick_elems, affine.buffer_aligned_page_size()]


def create_program_descriptor(
    input_tensor,
    gamma,
    beta,
    residual_input_tensor,
    output_tensor,
    blocking,
    *,
    epsilon,
    compute_kernel_config,
):
    cores = blocking.all_cores
    Wt = blocking.tensor_w_tiles
    block_height = blocking.block_height
    in_bytes = blocking.in_tile_bytes
    has_residual = blocking.has_residual
    width = list(input_tensor.shape)[-1]

    plan = blocking.cb_plan
    cbs = [
        _cb((CB_INPUT_TILES,), *plan["input"], input_tensor.dtype, cores),
        _cb((CB_GAMMA_ROWS,), *plan["gamma"], gamma.dtype, cores),
        _cb((CB_BETA_ROWS,), *plan["beta"], beta.dtype, cores),
        _cb(WORK_GROUP, *plan["work"], blocking.acc_dtype, cores),
        _cb(STATS_GROUP, *plan["stats"], blocking.acc_dtype, cores),
        _cb((CB_OUTPUT_TILES,), *plan["output"], output_tensor.dtype, cores),
    ]
    if has_residual:
        cbs.append(_cb((CB_RESIDUAL_TILES,), *plan["residual"], residual_input_tensor.dtype, cores))

    residual_args = (
        ttnn.TensorAccessorArgs(residual_input_tensor).get_compile_time_args()
        if has_residual
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    residual_on_writer = has_residual and blocking.residual_on_writer
    reader_ct_args = [CB_INPUT_TILES, Wt, in_bytes, CB_RESIDUAL_TILES, int(has_residual and not residual_on_writer)]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(residual_args)

    writer_ct_args = [
        CB_OUTPUT_TILES,
        CB_GAMMA_ROWS,
        CB_BETA_ROWS,
        CB_RESIDUAL_TILES,
        Wt,
        plan["output"][1],
        int(residual_on_writer),
        in_bytes,
        plan["residual"][0] // Wt if has_residual else 0,  # residual CB capacity in tile-rows
    ]
    writer_ct_args.extend(_affine_ct_args(gamma))
    writer_ct_args.extend(_affine_ct_args(beta))
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    writer_ct_args.extend(residual_args)

    compute_ct_args = [
        CB_INPUT_TILES,
        CB_RESIDUAL_TILES,
        CB_GAMMA_ROWS,
        CB_BETA_ROWS,
        CB_XSUM,
        CB_CENTERED,
        CB_NORMED,
        CB_SCALED,
        CB_MEAN,
        CB_SUMSQ,
        CB_RSTD,
        CB_OUTPUT_TILES,
        Wt,
        block_height,
        int(has_residual),
        width,
        _float_bits(1.0 / width),
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    res_addr = residual_input_tensor.buffer_address() if has_residual else 0
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address()
    beta_addr = beta.buffer_address()
    eps_bits = _float_bits(epsilon)
    for core, start_row, num_blocks, last_block_extent in blocking.assignment:
        core_rows = (num_blocks - 1) * block_height + last_block_extent
        reader_rt[core.x][core.y] = [in_addr, res_addr, start_row, core_rows]
        writer_rt[core.x][core.y] = [out_addr, gamma_addr, beta_addr, res_addr, start_row, core_rows]
        compute_rt[core.x][core.y] = [num_blocks, last_block_extent, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
