# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: regime R1 (row_resident), rows split across the compute grid."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# Blocking-model host constants (op_design.md, Work Distribution).
MIN_BLOCKS_PER_CORE = 2
W_CHUNK_FLOOR = 8
L1_HEADROOM_BYTES = 64 * 1024
GAMMA_DEPTH = 2
OUT_DEPTH = 2

FLOAT32_TILE_BYTES = ttnn.tile_size(ttnn.float32)
BFLOAT16_TILE_BYTES = ttnn.tile_size(ttnn.bfloat16)

CB_INPUT_TILES = 0
CB_SCALER = 1
CB_GAMMA_STICK = 2
CB_GAMMA_TILES = 3
CB_SUMSQ = 4
CB_RINV = 5
CB_X_NORMED = 6
CB_OUTPUT_TILES = 7


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Blocking:
    block_row_tiles: int  # R
    w_chunk_tiles: int  # C
    in_depth: int
    tensor_w_tiles: int  # Wt
    has_partial_w: int  # P
    tensor_row_tiles: int
    all_cores: object
    assignment: tuple  # ((core, row_start, core_row_tiles), ...)


def _geometry(input_tensor):
    shape = list(input_tensor.shape)
    width = shape[-1]
    height = shape[-2]
    leading = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return width, leading * _div_up(height, TILE), _div_up(width, TILE)


def _gamma_stick_bytes(width, gamma):
    return max(_div_up(width, TILE) * TILE * gamma.element_size(), gamma.buffer_aligned_page_size())


def _footprint(R, C, in_depth, *, Wt, P, xt, ot, gamma_stick_bytes, gt, has_gamma):
    """Closed-form per-core CB bytes; mirrors l1_ledger.md."""
    g = 1 if has_gamma else 0
    return (
        in_depth * R * Wt * xt  # cb_input_tiles
        + (1 + P) * BFLOAT16_TILE_BYTES  # cb_scaler
        + (1 + P) * R * FLOAT32_TILE_BYTES  # cb_sumsq
        + (1 + P) * R * FLOAT32_TILE_BYTES  # cb_rinv
        + g * gamma_stick_bytes  # cb_gamma_stick
        + g * GAMMA_DEPTH * C * gt  # cb_gamma_tiles
        + g * R * C * FLOAT32_TILE_BYTES  # cb_x_normed
        + OUT_DEPTH * C * ot  # cb_output_tiles
    )


def _split_rows(device, tensor_row_tiles):
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, group_1, group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, tensor_row_tiles, row_wise=True)
    assignment = []
    start = 0
    for group, per_core in ((group_1, rows_g1), (group_2, rows_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    return all_cores, tuple(assignment), rows_g1


def select_blocking(input_tensor, gamma):
    """Single source of every block knob (R, C, in_depth). Returns None when R1's predicate fails."""
    width, tensor_row_tiles, Wt = _geometry(input_tensor)
    P = 1 if width % TILE else 0
    has_gamma = gamma is not None
    all_cores, assignment, core_row_tiles_max = _split_rows(input_tensor.device(), tensor_row_tiles)

    sizes = dict(
        Wt=Wt,
        P=P,
        xt=input_tensor.tile.get_tile_size(input_tensor.dtype),
        ot=input_tensor.tile.get_tile_size(input_tensor.dtype),
        gamma_stick_bytes=_gamma_stick_bytes(width, gamma) if has_gamma else 0,
        gt=ttnn.tile_size(gamma.dtype) if has_gamma else 0,
        has_gamma=has_gamma,
    )
    # Allocator-managed L1 per core (CBs start at the allocator base, not at the HAL unreserved base).
    l1_bytes = ttnn.get_memory_view(input_tensor.device(), ttnn.BufferType.L1).total_bytes_per_bank
    budget = l1_bytes - L1_HEADROOM_BYTES

    def depth(R):
        return 2 if _div_up(core_row_tiles_max, R) >= 2 else 1

    def fits(R, C, d):
        return _footprint(R, C, d, **sizes) <= budget

    c_floor = min(Wt, W_CHUNK_FLOOR)
    R = _div_up(core_row_tiles_max, MIN_BLOCKS_PER_CORE)
    while R > 1 and not fits(R, c_floor, depth(R)):
        R -= 1

    in_depth = depth(R)
    C = None
    if fits(R, c_floor, in_depth):
        C = next(c for c in range(Wt, c_floor - 1, -1) if fits(R, c, in_depth))
    else:
        C = next((c for c in range(c_floor - 1, 0, -1) if fits(1, c, in_depth)), None)
        if C is None:
            in_depth = 1
            C = next((c for c in range(Wt, 0, -1) if fits(1, c, in_depth)), None)
        if C is None:
            return None

    return Blocking(
        block_row_tiles=R,
        w_chunk_tiles=C,
        in_depth=in_depth,
        tensor_w_tiles=Wt,
        has_partial_w=P,
        tensor_row_tiles=tensor_row_tiles,
        all_cores=all_cores,
        assignment=assignment,
    )


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cb(index, num_pages, page_size, data_format, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(input_tensor, gamma, output_tensor, blocking, *, epsilon, compute_kernel_config):
    width = list(input_tensor.shape)[-1]
    has_gamma = gamma is not None
    R = blocking.block_row_tiles
    C = blocking.w_chunk_tiles
    Wt = blocking.tensor_w_tiles
    P = blocking.has_partial_w
    cores = blocking.all_cores

    input_tile_bytes = input_tensor.tile.get_tile_size(input_tensor.dtype)
    output_tile_bytes = output_tensor.tile.get_tile_size(output_tensor.dtype)

    cbs = [
        _cb(CB_INPUT_TILES, blocking.in_depth * R * Wt, input_tile_bytes, input_tensor.dtype, cores),
        _cb(CB_SCALER, 1 + P, BFLOAT16_TILE_BYTES, ttnn.bfloat16, cores),
        _cb(CB_SUMSQ, (1 + P) * R, FLOAT32_TILE_BYTES, ttnn.float32, cores),
        # (1 + P) * R: with a partial last W tile the second reduce call re-pushes R_b tiles, and a ragged
        # last block must not wrap mid-window when N1 indexes the R_b rinv tiles.
        _cb(CB_RINV, (1 + P) * R, FLOAT32_TILE_BYTES, ttnn.float32, cores),
        _cb(CB_OUTPUT_TILES, OUT_DEPTH * C, output_tile_bytes, output_tensor.dtype, cores),
    ]
    gamma_stick_bytes = 0
    gamma_tile_bytes = 0
    if has_gamma:
        gamma_stick_bytes = _gamma_stick_bytes(width, gamma)
        gamma_tile_bytes = ttnn.tile_size(gamma.dtype)
        cbs += [
            _cb(CB_GAMMA_STICK, 1, gamma_stick_bytes, gamma.dtype, cores),
            _cb(CB_GAMMA_TILES, GAMMA_DEPTH * C, gamma_tile_bytes, gamma.dtype, cores),
            _cb(CB_X_NORMED, R * C, FLOAT32_TILE_BYTES, ttnn.float32, cores),
        ]

    reader_ct_args = [CB_INPUT_TILES, CB_SCALER, R, Wt, input_tile_bytes, width % TILE]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        CB_GAMMA_STICK,
        CB_GAMMA_TILES,
        R,
        Wt,
        C,
        int(has_gamma),
        output_tile_bytes,
        gamma.element_size() if has_gamma else 0,
        gamma.buffer_aligned_page_size() if has_gamma else 0,
        gamma_stick_bytes,
        GAMMA_DEPTH,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    compute_ct_args = [
        CB_INPUT_TILES,
        CB_SCALER,
        CB_GAMMA_TILES,
        CB_SUMSQ,
        CB_RINV,
        CB_X_NORMED,
        CB_OUTPUT_TILES,
        R,
        Wt,
        C,
        P,
        int(has_gamma),
        _float_bits(1.0 / width),
        _float_bits(epsilon),
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    for core, row_start, core_rows in blocking.assignment:
        reader_rt[core.x][core.y] = [in_addr, row_start, core_rows]
        writer_rt[core.x][core.y] = [out_addr, gamma_addr, row_start, core_rows]
        compute_rt[core.x][core.y] = [core_rows]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
