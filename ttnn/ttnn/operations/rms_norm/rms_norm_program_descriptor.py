# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for the RMS normalization operation.

Work distribution: single core (1x1 grid).
Data flow: two-pass per tile-row:
  Pass 1: read Wt input tiles -> square -> reduce_row -> mean(x^2)
  Pass 2: re-read Wt input tiles -> add eps + rsqrt -> broadcast multiply -> output
  Optional: gamma tilize + multiply
"""

import math
import struct
from pathlib import Path
from typing import Optional

import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_bfloat16_bits(val: float) -> int:
    """Convert a float to its bfloat16 representation (upper 16 bits of float32)."""
    float32_bits = struct.unpack(">I", struct.pack(">f", val))[0]
    return (float32_bits >> 16) & 0xFFFF


def _pack_two_bfloat16(val: float) -> int:
    """Pack a float as two bfloat16 values into a uint32: (bf16 << 16 | bf16)."""
    bf16_bits = _float_to_bfloat16_bits(val)
    return (bf16_bits << 16) | bf16_bits


def _float_to_uint32(val: float) -> int:
    """Reinterpret a float32 as uint32 (raw bit pattern)."""
    return struct.unpack(">I", struct.pack(">f", val))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the RMS Norm operation.

    Args:
        input_tensor: Input tensor (on device)
        output_tensor: Pre-allocated output tensor (on device)
        gamma: Optional gamma/weight tensor (on device), shape (1,1,1,W) in RM layout
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    input_shape = input_tensor.shape
    rank = len(input_shape)

    W = input_shape[-1]
    H = input_shape[-2]

    # Batch dimensions: product of all dims except last two
    NC = 1
    for i in range(rank - 2):
        NC *= input_shape[i]

    is_rm_input = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    is_rm_output = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None

    # Tile dimensions (always 32x32 for standard tiles)
    TILE_H = 32
    TILE_W = 32

    # Width in tiles and height in tiles
    Wt = (W + TILE_W - 1) // TILE_W  # ceiling division
    Ht = (H + TILE_H - 1) // TILE_H

    # Total tile-rows to process
    num_rows = NC * Ht

    # Tile size in bytes (for the input data format)
    tile_size = ttnn.tile_size(input_tensor.dtype)

    # Stick size for RM layout (full row width in bytes)
    elem_size = input_tensor.element_size()
    stick_size = W * elem_size
    output_W = output_tensor.shape[-1]
    Wt_out = (output_W + TILE_W - 1) // TILE_W
    output_stick_size = output_W * output_tensor.element_size()

    # Gamma stick size (if present, gamma is always RM with shape (1,1,1,W))
    gamma_stick_size = 0
    if has_gamma:
        gamma_stick_size = W * gamma.element_size()

    # Total tiles and sticks for the input
    num_tiles = input_tensor.buffer_num_pages() if not is_rm_input else 0
    num_sticks = input_tensor.buffer_num_pages() if is_rm_input else 0

    # Scaler: 1/W packed as two bfloat16 values in a uint32
    # The reduce LLK requires bfloat16 scaler regardless of input dtype
    scaler_val = 1.0 / float(W)
    packed_scaler = _pack_two_bfloat16(scaler_val)

    # Epsilon packed as uint32 (raw float32 bits)
    packed_eps = _float_to_uint32(epsilon)

    # ========== 2. CORE GRID (SINGLE CORE) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # All tile-layout CBs use tile_size page size.
    # RM CBs also use tile_size page size (32 sticks packed as tile-equivalent for tilize).
    # cb_scaler always uses bfloat16 regardless of input dtype.

    cbs = []

    # --- c_0: cb_in_rm (RM sticks for tilize, only for RM input) ---
    if is_rm_input:
        cb0 = ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
        cbs.append(cb0)

    # --- c_1: cb_in (tilized / tile input) ---
    cb1 = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cbs.append(cb1)

    # --- c_2: cb_x_sq (x^2 intermediate, Wt pages needed because square fills all
    #     tiles before reduce consumes them -- both run on same compute RISC) ---
    cb2 = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cbs.append(cb2)

    # --- c_3: cb_gamma_rm (gamma RM sticks for tilize, only when gamma present) ---
    if has_gamma:
        cb3 = ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=3,
                    data_format=gamma.dtype,
                    page_size=tile_size,
                )
            ],
        )
        cbs.append(cb3)

    # --- c_4: cb_gamma (tilized gamma, 2 pages for streaming) ---
    if has_gamma:
        cb4 = ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=4,
                    data_format=gamma.dtype if has_gamma else input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
        cbs.append(cb4)

    # --- c_8: cb_scaler (reduce scaler 1/W, always bfloat16, 1 page) ---
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)
    cb8 = ttnn.CBDescriptor(
        total_size=1 * scaler_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=8,
                data_format=ttnn.bfloat16,
                page_size=scaler_tile_size,
            )
        ],
    )
    cbs.append(cb8)

    # --- c_9: cb_eps (epsilon tile, uses input data format, 1 page) ---
    cb9 = ttnn.CBDescriptor(
        total_size=1 * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=9,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cbs.append(cb9)

    # --- c_16: cb_out (output tiled data) ---
    # For RM output: Wt_out pages (untilize needs all output tiles accumulated)
    # For TILE output: 2 pages (double buffer streaming)
    cb16_pages = Wt_out if is_rm_output else 2
    cb16 = ttnn.CBDescriptor(
        total_size=cb16_pages * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=16,
                data_format=output_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cbs.append(cb16)

    # --- c_17: cb_out_rm (untilized output sticks, only for RM output) ---
    if is_rm_output:
        cb17 = ttnn.CBDescriptor(
            total_size=Wt_out * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=17,
                    data_format=output_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
        cbs.append(cb17)

    # --- c_24: cb_reduce_out (mean(x^2) accumulator, 2 pages) ---
    cb24 = ttnn.CBDescriptor(
        total_size=2 * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=24,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cbs.append(cb24)

    # --- c_25: cb_rms_inv (rsqrt(mean+eps), 2 pages) ---
    cb25 = ttnn.CBDescriptor(
        total_size=2 * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=25,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )
    cbs.append(cb25)

    # --- c_26: cb_norm (normalized output pre-gamma, only when gamma present) ---
    if has_gamma:
        cb26 = ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=26,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
        cbs.append(cb26)

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader Kernel ---
    # CT args: is_rm_input, has_gamma, stick_size, gamma_stick_size,
    #          TensorAccessorArgs(input), [TensorAccessorArgs(gamma) or placeholder]
    reader_ct_args = [
        int(is_rm_input),
        int(has_gamma),
        stick_size,
        gamma_stick_size,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.append(0)  # placeholder for absent gamma

    # RT args: src_addr, gamma_addr, num_rows, Wt, num_sticks, num_tiles, packed_scaler, packed_eps
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address() if has_gamma else 0,
        num_rows,
        Wt,
        num_sticks,
        num_tiles if not is_rm_input else 0,
        packed_scaler,
        packed_eps,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute Kernel ---
    # CT args: is_rm_input, has_gamma, Wt, Ht, NC
    fp32_acc = input_tensor.dtype == ttnn.float32
    compute_ct_args = [
        int(is_rm_input),
        int(has_gamma),
        Wt,
        Ht,
        NC,
    ]

    # RT args: num_rows
    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = [num_rows]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=fp32_acc,
            math_approx_mode=False,
        ),
    )

    # --- Writer Kernel ---
    # CT args: is_rm_output, output_stick_size, TensorAccessorArgs(output)
    writer_ct_args = [
        int(is_rm_output),
        output_stick_size,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # RT args: dst_addr, num_rows, Wt_out, num_tiles (total output tiles or sticks)
    total_output_pages = output_tensor.buffer_num_pages()
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_rows,
        Wt_out,
        total_output_pages,
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )
