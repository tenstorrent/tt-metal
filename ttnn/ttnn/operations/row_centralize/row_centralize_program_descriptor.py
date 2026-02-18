# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Centralize - Program Descriptor

Single-core implementation:
  - Reader reads RM sticks from DRAM into CB c_0, also generates scaler and epsilon tiles once
  - If affine: reader reads gamma/beta sticks into c_0 at startup (before main loop)
  - Compute: tilize -> reduce_mean -> sub -> square -> reduce_var -> add_eps -> rsqrt -> mul
             -> [if affine: mul_gamma -> add_beta] -> untilize
  - Writer writes RM sticks from CB c_16 to DRAM

CB layout:
  c_0  (cb_rm_in)       - input RM sticks from reader (reused for gamma/beta at startup)
  c_1  (cb_tilized)     - tilized input tiles
  c_2  (cb_mean)        - row mean tile
  c_3  (cb_centered)    - centered tiles (x - mean), persistent across square + mul
  c_4  (cb_var)         - variance tile
  c_5  (cb_inv_std)     - 1/sqrt(var + eps) tile
  c_6  (cb_result)      - standardized tiles (TILE fmt, untilize input)
  c_7  (cb_eps)         - epsilon scalar tile (program lifetime)
  c_8  (cb_scaler)      - reduce scaler tile 1/W (program lifetime)
  c_9  (cb_gamma_tiled) - tilized gamma (program lifetime, only when has_affine)
  c_10 (cb_beta_tiled)  - tilized beta (program lifetime, only when has_affine)
  c_16 (cb_rm_out)      - output RM sticks for writer
  c_24 (cb_squared)     - squared centered tiles
  c_25 (cb_var_plus_eps)- var + epsilon intermediate
  c_26 (cb_after_gamma) - intermediate: standardized * gamma (only when has_affine)
"""

import struct
from pathlib import Path
import ttnn

# Kernel files relative to tt-metal repo root (required by generic_op kernel_source)
KERNEL_BASE = "ttnn/ttnn/operations/row_centralize/kernels"

# Tile height/width (always 32 for Tenstorrent hardware)
TILE_H = 32
TILE_W = 32

# CB IDs matching spec
CB_RM_IN = 0
CB_TILIZED = 1
CB_MEAN = 2
CB_CENTERED = 3
CB_VAR = 4
CB_INV_STD = 5
CB_RESULT = 6
CB_EPS = 7
CB_SCALER = 8
CB_RM_OUT = 16
CB_GAMMA_TILED = 9
CB_BETA_TILED = 10
CB_SQUARED = 24
CB_VAR_PLUS_EPS = 25
CB_AFTER_GAMMA = 26


def _float_to_packed_bf16(value: float) -> int:
    """
    Convert a float to a packed bf16 value: (bf16 << 16 | bf16).

    This is the format expected by generate_reduce_scaler() and
    generate_bcast_scalar_bfloat16() in the kernel helpers.

    Note: bf16 is the upper 16 bits of the IEEE 754 float32 representation,
    obtained by truncating (rounding towards zero) the mantissa.
    """
    # Pack float32 to 4 bytes, take upper 2 bytes as bf16
    float32_bytes = struct.pack(">f", value)  # big-endian float32
    bf16 = (float32_bytes[0] << 8) | float32_bytes[1]  # upper 16 bits
    # Pack as (bf16 << 16 | bf16)
    return (bf16 << 16) | bf16


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    epsilon: float = 1e-5,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for row_centralize.

    Args:
        input_tensor: Input tensor (RM, bf16, interleaved, on device)
        output_tensor: Pre-allocated output tensor (RM, bf16, interleaved, on device)
        epsilon: Numerical stability constant for rsqrt
        gamma: Optional scale tensor (RM, bf16, interleaved, shape [1,...,1,W])
        beta: Optional bias tensor (RM, bf16, interleaved, shape [1,...,1,W])

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    has_affine = gamma is not None
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)

    # Width (last dim) and height (second-to-last dim)
    W = shape[rank - 1]
    H = shape[rank - 2]

    # Tiles per row and total tile-row count
    Wt = W // TILE_W  # tiles per row-group (tile-row width)
    Ht = H // TILE_H  # tile-rows per H dimension

    # Total tile-rows across all batch dims and H dimension
    # For a shape [B, H, W], Ht_total = B * (H/32)
    # For a shape [B, C, H, W], Ht_total = B * C * (H/32)
    total_volume = 1
    for i in range(rank):
        total_volume *= shape[i]
    # Total elements / (H * W) = product of all batch dims
    # Then Ht_total = batch_dims_product * Ht
    # Equivalently: total_volume / (W * TILE_H)
    Ht_total = total_volume // (W * TILE_H)

    # Stick size: one row of elements in bytes (bf16 = 2 bytes per element)
    stick_size = W * 2  # bytes

    # Tile size for BF16: 32 * 32 * 2 bytes = 2048 bytes
    tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Total sticks: each tile-row has TILE_H = 32 sticks
    num_sticks = Ht_total * TILE_H

    # Reduce scaler: 1/W packed as (bf16 << 16 | bf16)
    reduce_scaler_value = 1.0 / float(W)
    packed_reduce_scaler = _float_to_packed_bf16(reduce_scaler_value)

    # Epsilon packed as (bf16 << 16 | bf16)
    packed_eps = _float_to_packed_bf16(epsilon)

    # ========== 2. CORE GRID (SINGLE CORE) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # Wt-tile CBs: capacity for one full tile-row
    Wt_total_size = Wt * tile_size
    # Single-tile CBs
    single_tile_size = tile_size

    # c_0 (cb_rm_in): input RM sticks - capacity = Wt tiles worth of RM data
    # RM sticks for one tile-row: Wt * TILE_W elements * 2 bytes = Wt * stick_size bytes
    # But CB page_size must match what the reader writes: one stick = stick_size bytes
    # Total size = TILE_H sticks per tile-row = 32 sticks = Wt * tile_size bytes
    # (because TILE_H * stick_size = 32 * W * 2 = Wt * 32 * 32 * 2 = Wt * tile_size)
    cb_rm_in_descriptor = ttnn.CBDescriptor(
        total_size=Wt_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RM_IN,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_1 (cb_tilized): tilized tiles - Wt tiles capacity
    cb_tilized_descriptor = ttnn.CBDescriptor(
        total_size=Wt_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TILIZED,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_2 (cb_mean): 1 tile
    cb_mean_descriptor = ttnn.CBDescriptor(
        total_size=single_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_3 (cb_centered): Wt tiles, persistent across square and mul phases
    cb_centered_descriptor = ttnn.CBDescriptor(
        total_size=Wt_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_CENTERED,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_4 (cb_var): 1 tile
    cb_var_descriptor = ttnn.CBDescriptor(
        total_size=single_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_VAR,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_5 (cb_inv_std): 1 tile
    cb_inv_std_descriptor = ttnn.CBDescriptor(
        total_size=single_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INV_STD,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_6 (cb_result): Wt tiles - standardized tiles before untilize
    cb_result_descriptor = ttnn.CBDescriptor(
        total_size=Wt_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RESULT,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_7 (cb_eps): 1 tile, program lifetime
    cb_eps_descriptor = ttnn.CBDescriptor(
        total_size=single_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EPS,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_8 (cb_scaler): 1 tile, program lifetime
    cb_scaler_descriptor = ttnn.CBDescriptor(
        total_size=single_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_16 (cb_rm_out): Wt tiles, output RM sticks for writer
    cb_rm_out_descriptor = ttnn.CBDescriptor(
        total_size=Wt_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RM_OUT,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_24 (cb_squared): Wt tiles
    cb_squared_descriptor = ttnn.CBDescriptor(
        total_size=Wt_total_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SQUARED,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # c_25 (cb_var_plus_eps): 1 tile
    cb_var_plus_eps_descriptor = ttnn.CBDescriptor(
        total_size=single_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_VAR_PLUS_EPS,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    all_cbs = [
        cb_rm_in_descriptor,
        cb_tilized_descriptor,
        cb_mean_descriptor,
        cb_centered_descriptor,
        cb_var_descriptor,
        cb_inv_std_descriptor,
        cb_result_descriptor,
        cb_eps_descriptor,
        cb_scaler_descriptor,
        cb_rm_out_descriptor,
        cb_squared_descriptor,
        cb_var_plus_eps_descriptor,
    ]

    # Affine CBs (only allocated when has_affine)
    if has_affine:
        # c_9 (cb_gamma_tiled): Wt tiles, program lifetime — tilized gamma
        cb_gamma_tiled_descriptor = ttnn.CBDescriptor(
            total_size=Wt_total_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_GAMMA_TILED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )

        # c_10 (cb_beta_tiled): Wt tiles, program lifetime — tilized beta
        cb_beta_tiled_descriptor = ttnn.CBDescriptor(
            total_size=Wt_total_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_BETA_TILED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )

        # c_26 (cb_after_gamma): Wt tiles, per tile-row — intermediate: standardized * gamma
        cb_after_gamma_descriptor = ttnn.CBDescriptor(
            total_size=Wt_total_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_AFTER_GAMMA,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )

        all_cbs.extend([cb_gamma_tiled_descriptor, cb_beta_tiled_descriptor, cb_after_gamma_descriptor])

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader Kernel ---
    # Compile-time args:
    #   [0] stick_size, [1] cb_rm_in, [2] cb_scaler, [3] cb_eps,
    #   [4] has_affine, [5+] TensorAccessorArgs(src),
    #   [next+] TensorAccessorArgs(gamma), [next+] TensorAccessorArgs(beta)
    reader_ct_args = [
        stick_size,  # [0] stick_size: W * 2 bytes
        CB_RM_IN,  # [1] cb_rm_in = c_0
        CB_SCALER,  # [2] cb_scaler = c_8
        CB_EPS,  # [3] cb_eps = c_7
        1 if has_affine else 0,  # [4] has_affine flag
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Always include gamma/beta TensorAccessorArgs (use input_tensor as dummy when no affine).
    # This is required because TensorAccessorArgs template instantiation in the kernel happens
    # at compile time even inside `if constexpr (false)` in non-template kernel_main().
    gamma_tensor = gamma if has_affine else input_tensor
    beta_tensor = beta if has_affine else input_tensor
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args())

    # Runtime args:
    #   [0] src_addr, [1] num_sticks, [2] Wt, [3] start_stick_id,
    #   [4] packed_reduce_scaler, [5] packed_eps,
    #   [6] gamma_addr (0 when no affine), [7] beta_addr (0 when no affine)
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),  # [0] src_addr
        num_sticks,  # [1] num_sticks = Ht_total * 32
        Wt,  # [2] Wt (tiles per tile-row)
        0,  # [3] start_stick_id (single-core, start at 0)
        packed_reduce_scaler,  # [4] 1/W as packed bf16
        packed_eps,  # [5] epsilon as packed bf16
        gamma.buffer_address() if has_affine else 0,  # [6] gamma_addr
        beta.buffer_address() if has_affine else 0,  # [7] beta_addr
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_BASE}/row_centralize_reader.cpp",
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer Kernel ---
    # Compile-time args per spec:
    #   [0] cb_rm_out, [1] output_stick_size, [2] tile_height, [3] Wt, [4+] TensorAccessorArgs
    writer_ct_args = [
        CB_RM_OUT,  # [0] cb_rm_out = c_16
        stick_size,  # [1] output_stick_size: W * 2 bytes
        TILE_H,  # [2] tile_height = 32
        Wt,  # [3] Wt
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Runtime args per spec:
    #   [0] dst_addr, [1] num_tile_rows, [2] start_tile_row
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),  # [0] dst_addr
        Ht_total,  # [1] num_tile_rows
        0,  # [2] start_tile_row (single-core)
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_BASE}/row_centralize_writer.cpp",
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute Kernel ---
    # Compile-time args:
    #   [0] Wt, [1] Ht_total, [2..13] CB IDs,
    #   [14] cb_gamma_tiled, [15] cb_beta_tiled, [16] cb_after_gamma, [17] has_affine
    compute_ct_args = [
        Wt,  # [0] Wt
        Ht_total,  # [1] Ht_total
        CB_RM_IN,  # [2] cb_rm_in = c_0
        CB_TILIZED,  # [3] cb_tilized = c_1
        CB_MEAN,  # [4] cb_mean = c_2
        CB_CENTERED,  # [5] cb_centered = c_3
        CB_SQUARED,  # [6] cb_squared = c_24
        CB_VAR,  # [7] cb_var = c_4
        CB_VAR_PLUS_EPS,  # [8] cb_var_plus_eps = c_25
        CB_INV_STD,  # [9] cb_inv_std = c_5
        CB_RESULT,  # [10] cb_result = c_6
        CB_RM_OUT,  # [11] cb_rm_out = c_16
        CB_EPS,  # [12] cb_eps = c_7
        CB_SCALER,  # [13] cb_scaler = c_8
        CB_GAMMA_TILED if has_affine else 0,  # [14] cb_gamma_tiled
        CB_BETA_TILED if has_affine else 0,  # [15] cb_beta_tiled
        CB_AFTER_GAMMA if has_affine else 0,  # [16] cb_after_gamma
        1 if has_affine else 0,  # [17] has_affine
    ]

    # Compute has no runtime args (all single-core, all parameters are compile-time)
    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = []

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_BASE}/row_centralize_compute.cpp",
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 5. ASSEMBLE PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
