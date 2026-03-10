# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Group Norm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.

CB Layout:
  CB 0  (cb_input_rm)      : RM sticks packed as tile-sized pages, Ct pages per tile-row
  CB 1  (cb_tilized)       : Persistent tilized input, Ht*Ct pages per sample
  CB 2  (cb_gamma)         : Gamma tiles (TILE_LAYOUT), Ct pages (persistent)
  CB 3  (cb_beta)          : Beta tiles (TILE_LAYOUT), Ct pages (persistent)
  CB 4  (cb_eps)           : Epsilon scalar broadcast tile, 1 page (persistent)
  CB 5  (cb_scaler)        : Reduce scaler tile (value 1/K), 1 page (persistent)
  CB 6  (cb_mean)          : Group mean scalar tile, G pages (all groups stored)
  CB 7  (cb_den)           : Group rsqrt(var+eps) scalar tile, 1 page per group
  CB 16 (cb_normalized)    : Normalized output tiles, Ct pages per tile-row
  CB 17 (cb_output_rm)     : Untilized RM data, Ct pages per tile-row
  CB 24 (cb_sq_sum)        : E[x^2] accumulator, 1 page per group
  CB 25 (cb_tmp)           : Scratch (squared tile staging), 1 page per tile
  CB 26 (cb_group_scaler)  : Per-group scaler mask tiles, G*Ct pages (persistent)
"""

import struct
from pathlib import Path

import ttnn


def _bf16_packed_scalar(value: float) -> int:
    """Convert a float to double-packed bf16 uint32 for generate_reduce_scaler.

    Truncates the float to bf16 by dropping the lower 16 bits of the IEEE 754
    float32 representation, then packs the bf16 value into both halves of a uint32.
    """
    f32_bits = struct.unpack("I", struct.pack("f", value))[0]
    bf16_val = f32_bits >> 16  # truncate to bf16
    return (bf16_val << 16) | bf16_val


# Kernel files are in the kernels/ subdirectory relative to this file
KERNEL_DIR = Path(__file__).parent / "kernels"

# CB index constants
CB_INPUT_RM = 0
CB_TILIZED = 1
CB_GAMMA = 2
CB_BETA = 3
CB_EPS = 4
CB_SCALER = 5
CB_MEAN = 6
CB_DEN = 7
CB_NORMALIZED = 16
CB_OUTPUT_RM = 17
CB_SQ_SUM = 24
CB_TMP = 25
CB_GROUP_SCALER = 26


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor,
    beta_tensor: ttnn.Tensor,
    group_scaler_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_groups: int = 1,
    eps_packed: int = 0,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the group_norm operation.
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    N = shape[0]
    HW = shape[2]
    C = shape[3]

    Ht = HW // 32
    Ct = C // 32
    G = num_groups

    stick_size = input_tensor.buffer_page_size()
    tile_page_size = gamma_tensor.buffer_page_size()
    output_stick_size = output_tensor.buffer_page_size()
    group_scaler_tile_page_size = group_scaler_tensor.buffer_page_size()

    # ========== 2. CORE GRID (single core) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    cb_input_rm_desc = ttnn.CBDescriptor(
        total_size=Ct * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_RM,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_tilized_desc = ttnn.CBDescriptor(
        total_size=Ht * Ct * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TILIZED,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_gamma_desc = ttnn.CBDescriptor(
        total_size=Ct * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA,
                data_format=gamma_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_beta_desc = ttnn.CBDescriptor(
        total_size=Ct * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA,
                data_format=beta_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_eps_desc = ttnn.CBDescriptor(
        total_size=1 * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EPS,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_scaler_desc = ttnn.CBDescriptor(
        total_size=1 * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    # CB 6: cb_mean - G pages to store all group means
    cb_mean_desc = ttnn.CBDescriptor(
        total_size=G * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_den_desc = ttnn.CBDescriptor(
        total_size=G * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_DEN,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_normalized_desc = ttnn.CBDescriptor(
        total_size=Ct * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_NORMALIZED,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_output_rm_desc = ttnn.CBDescriptor(
        total_size=Ct * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_RM,
                data_format=output_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_sq_sum_desc = ttnn.CBDescriptor(
        total_size=1 * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SQ_SUM,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    cb_tmp_desc = ttnn.CBDescriptor(
        total_size=1 * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TMP,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    # CB 26: cb_group_scaler - G*Ct persistent tiles of per-group scaler masks
    cb_group_scaler_desc = ttnn.CBDescriptor(
        total_size=G * Ct * group_scaler_tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GROUP_SCALER,
                data_format=group_scaler_tensor.dtype,
                page_size=group_scaler_tile_page_size,
            )
        ],
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(group_scaler_tensor).get_compile_time_args())

    num_sticks = N * HW

    import math

    channels_per_group = C // G
    K = HW * channels_per_group  # elements per group per sample
    # REDUCE_SCALAR applies the scaler at both row and column reduction stages,
    # so effective scaler = scaler^2.  We need effective = 1/K, so pass 1/sqrt(K).
    inv_K_packed = _bf16_packed_scalar(1.0 / math.sqrt(K))
    # Convert eps from float32-packed to double-packed bf16 for generate_reduce_scaler
    eps_float = struct.unpack("f", struct.pack("I", eps_packed))[0]
    eps_bf16_packed = _bf16_packed_scalar(eps_float)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_tensor.buffer_address(),
        beta_tensor.buffer_address(),
        group_scaler_tensor.buffer_address(),
        num_sticks,
        Ct,
        G * Ct,  # group_scaler_num_tiles
        inv_K_packed,  # pre-packed bf16 1/K for reduce scaler
        eps_bf16_packed,  # pre-packed bf16 eps for eps scaler tile
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "group_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [Ht, Ct, G, N]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "group_norm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [CB_OUTPUT_RM, output_stick_size, 32, N * Ht, Ct]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "group_norm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[
            cb_input_rm_desc,
            cb_tilized_desc,
            cb_gamma_desc,
            cb_beta_desc,
            cb_eps_desc,
            cb_scaler_desc,
            cb_mean_desc,
            cb_den_desc,
            cb_normalized_desc,
            cb_output_rm_desc,
            cb_sq_sum_desc,
            cb_tmp_desc,
            cb_group_scaler_desc,
        ],
    )
