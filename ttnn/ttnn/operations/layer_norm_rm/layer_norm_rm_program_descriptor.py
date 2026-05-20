# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for layer_norm_rm.

Single-core, per-tile-row processing of a (..., H, W) input.  Each of the
`total_tile_rows = ceil(prod(shape[:-1]) / 32)` iterations runs three passes:

    Pass 1: stream input  → mean   (reduce<SUM, REDUCE_ROW>, scaler 1/W)
    Pass 2: stream input  → variance via sub<COL>+square_in_place+reduce,
                            then transform_in_place to rsqrt(var+eps)
    Pass 3: stream input  → centered = (x-mean) * inv_std (* gamma) (+ beta)
                            drain via copy_tiles (TILE out) or untilize (RM out)

ROW_MAJOR input streams sticks (read_sticks_for_tilize<ROW>) and compute calls
tilize asymmetric (num_blocks=1, total_input_pages=32) once per pass per row.
TILE input streams tiles directly into cb_input_tiles (no in-kernel tilize).

Gamma/beta (if present) are always ROW_MAJOR sticks of length W; the reader reads
one stick into cb_gamma_sticks / cb_beta_sticks, and the compute kernel tilizes
each once into cb_gamma_tiles / cb_beta_tiles where they persist for every row's
Pass 3.

The scaler CB holds either a single full-tile scaler (W tile-aligned) or a
full+partial scaler pair (W non-tile-aligned). reduce<> waits but never pops; the
kernel issues a final cb_pop_front at exit.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def _padded_row_bytes(W: int, elem_size: int) -> int:
    tile_row_bytes = TILE_DIM * elem_size
    return math.ceil((W * elem_size) / tile_row_bytes) * tile_row_bytes


def _bits_of_float(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor | None,
    beta_tensor: ttnn.Tensor | None,
    output_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
) -> ttnn.ProgramDescriptor:
    # ----- Tensor metadata -----
    input_shape = list(input_tensor.shape)
    origin_W = int(input_shape[-1])
    elem_size = input_tensor.element_size()
    tile_size = ttnn.tile_size(input_tensor.dtype)

    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    # total_tile_rows = ceil(prod(shape[:-1]) / 32). prod over all leading dims
    # (including the inner H), flattened, padded up to a multiple of TILE_DIM.
    total_rows = 1
    for d in input_shape[:-1]:
        total_rows *= int(d)
    total_tile_rows = (total_rows + TILE_DIM - 1) // TILE_DIM

    is_rm_input = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    is_rm_output = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma_tensor is not None
    has_beta = beta_tensor is not None

    padded_row_bytes = _padded_row_bytes(origin_W, elem_size)
    input_row_bytes = origin_W * elem_size  # actual valid bytes per row in DRAM

    # Pages (= tiles for TILE layout, = sticks for RM layout) of each tensor.
    if is_rm_input:
        input_total_pages = total_rows  # sticks
    else:
        input_total_pages = total_tile_rows * Wt  # tiles

    if is_rm_output:
        output_total_pages = total_rows
    else:
        output_total_pages = total_tile_rows * Wt

    inv_W = 1.0 / float(origin_W)
    inv_W_bits = _bits_of_float(inv_W)
    eps_bits = _bits_of_float(epsilon)

    # ----- Core grid (single core) -----
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ----- Circular buffer indices -----
    CB_INPUT_STICKS = 0  # RM input only
    CB_INPUT_TILES = 1
    CB_GAMMA_STICKS = 2  # HAS_GAMMA only
    CB_BETA_STICKS = 3  # HAS_BETA only
    CB_SCALER = 4
    CB_OUTPUT = 16
    CB_GAMMA_TILES = 24  # HAS_GAMMA only
    CB_BETA_TILES = 25  # HAS_BETA only
    CB_MEAN = 26
    CB_INV_STD = 27
    CB_CENTERED = 28

    cbs: list[ttnn.CBDescriptor] = []

    # cb_input_tiles — Wt tiles (single-buffer; RM path: compute produces+consumes
    # sequentially; TILE path: reader pushes Wt at a time, compute consumes Wt).
    cb_input_tiles_pages = 2 * Wt if not is_rm_input else Wt
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb_input_tiles_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    if is_rm_input:
        # RM input: 32 sticks per tile-row (single-buffer; one tile-row at a time).
        cbs.append(
            ttnn.CBDescriptor(
                total_size=TILE_DIM * padded_row_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_STICKS,
                        data_format=input_tensor.dtype,
                        page_size=padded_row_bytes,
                    )
                ],
            )
        )

    # cb_scaler — bfloat16 ALWAYS (per reduce_helpers_dataflow.hpp).
    cb_scaler_pages = 2 if has_partial_w else 1
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb_scaler_pages * scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # cb_output — page_size is ALWAYS tile_size. The untilize helper outputs
    # tile-sized pages (Wt per tile-row) where each tile holds 32 rows of
    # padded_row_bytes laid out in row-major order. For RM output, the writer
    # uses dataflow_kernel_lib::write_sticks_after_untilize to extract sticks.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT,
                    data_format=output_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_mean, cb_inv_std — 1 tile each (per-row scalars, held by helpers across passes).
    for cb_idx in (CB_MEAN, CB_INV_STD):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_idx,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # cb_centered — Wt tiles, sized for one full row's in-place chain.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    if has_gamma:
        # cb_gamma_sticks — 32 pages of padded_row_bytes so the tilize LLK has
        # enough L1 to read 32 rows even though we only push 1 stick.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=TILE_DIM * padded_row_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_STICKS,
                        data_format=input_tensor.dtype,
                        page_size=padded_row_bytes,
                    )
                ],
            )
        )
        # cb_gamma_tiles — Wt tiles, persistent for every row's Pass 3.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_TILES,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=TILE_DIM * padded_row_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_STICKS,
                        data_format=input_tensor.dtype,
                        page_size=padded_row_bytes,
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_TILES,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # ----- Reader kernel -----
    # CT args: scalar args first, then TensorAccessorArgs at the end (input, gamma, beta).
    # gamma/beta accessor args are always present — use no-arg placeholder when absent.
    reader_ct_args: list[int] = [
        Wt,  # 0
        total_tile_rows,  # 1
        padded_row_bytes,  # 2
        input_row_bytes,  # 3
        inv_W_bits,  # 4
        1 if has_partial_w else 0,  # 5
        partial_w if has_partial_w else 0,  # 6
        1 if is_rm_input else 0,  # 7
        1 if has_gamma else 0,  # 8
        1 if has_beta else 0,  # 9
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args()
        if gamma_tensor is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args()
        if beta_tensor is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_tensor.buffer_address() if gamma_tensor is not None else 0,
        beta_tensor.buffer_address() if beta_tensor is not None else 0,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- Writer kernel -----
    writer_ct_args: list[int] = [
        Wt,  # 0
        total_tile_rows,  # 1
        padded_row_bytes,  # 2
        input_row_bytes,  # 3 (output row_bytes == input row_bytes)
        1 if is_rm_output else 0,  # 4
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address()]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ----- Compute kernel -----
    compute_ct_args: list[int] = [
        Wt,  # 0
        total_tile_rows,  # 1
        1 if has_partial_w else 0,  # 2
        1 if is_rm_input else 0,  # 3
        1 if is_rm_output else 0,  # 4
        1 if has_gamma else 0,  # 5
        1 if has_beta else 0,  # 6
        eps_bits,  # 7
    ]

    compute_config = compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor()

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
