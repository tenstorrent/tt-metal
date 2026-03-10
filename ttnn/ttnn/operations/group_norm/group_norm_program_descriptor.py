# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Group Norm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.

CB Layout (from op_design.md):
  CB 0  (cb_input_rm)   : RM sticks packed as tile-sized pages, Ct pages per tile-row
  CB 1  (cb_tilized)    : Persistent tilized input, Ht*Ct pages per sample
  CB 2  (cb_gamma)      : Gamma tiles (TILE_LAYOUT), Ct pages (persistent)
  CB 3  (cb_beta)       : Beta tiles (TILE_LAYOUT), Ct pages (persistent)
  CB 4  (cb_eps)        : Epsilon scalar broadcast tile, 1 page (persistent)
  CB 5  (cb_scaler)     : 1/K scaler tile, 1 page (persistent)
  CB 6  (cb_mean)       : Group mean scalar tile, 1 page per group
  CB 7  (cb_den)        : Group rsqrt(var+eps) scalar tile, 1 page per group
  CB 16 (cb_normalized) : Normalized output tiles, Ct pages per tile-row
  CB 17 (cb_output_rm)  : Untilized RM data, Ct pages per tile-row
  CB 24 (cb_sq_sum)     : E[x^2] accumulator, 1 page per group
  CB 25 (cb_tmp)        : Scratch (squared tile staging), 1 page per tile
"""

from pathlib import Path

import ttnn


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


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor,
    beta_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_groups: int = 1,
    eps_packed: int = 0,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the group_norm operation.

    Args:
        input_tensor: Input tensor (N, 1, H*W, C) ROW_MAJOR on device
        gamma_tensor: Gamma tensor (1, 1, 32, C) TILE_LAYOUT on device
        beta_tensor: Beta tensor (1, 1, 32, C) TILE_LAYOUT on device
        output_tensor: Pre-allocated output tensor (N, 1, H*W, C) ROW_MAJOR on device
        num_groups: Number of groups G
        eps_packed: Epsilon as packed uint32 float

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    N = shape[0]
    HW = shape[2]
    C = shape[3]

    Ht = HW // 32  # tile-rows per sample
    Ct = C // 32  # tile-columns
    Ct_g = Ct // num_groups  # tile-columns per group
    G = num_groups

    # For RM tensors: page = one stick = C * element_size bytes
    # stick_size = C * 2  (bfloat16)
    stick_size = input_tensor.buffer_page_size()

    # For tile tensors: page = one 32x32 tile
    tile_page_size = gamma_tensor.buffer_page_size()

    # Output stick size (same as input for same-shape RM output)
    output_stick_size = output_tensor.buffer_page_size()

    # ========== 2. CORE GRID (single core) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # All tile-sized pages use the same page size
    # For bf16 32x32 tiles: 2048 bytes

    # CB 0: cb_input_rm - RM sticks packed as tile-sized pages
    # The reader packs 32 sticks (each stick_size bytes) into Ct tile-sized pages
    # per tile-row. Each tile-page holds the data for 32 rows of 32 columns.
    # Page size for this CB is tile_page_size (so tilize helper works correctly).
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

    # CB 1: cb_tilized - Persistent buffer for tilized input (Ht*Ct tiles per sample)
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

    # CB 2: cb_gamma - Gamma tiles (persistent, Ct tiles)
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

    # CB 3: cb_beta - Beta tiles (persistent, Ct tiles)
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

    # CB 4: cb_eps - Epsilon scalar broadcast tile (1 tile, persistent)
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

    # CB 5: cb_scaler - 1/K scaler tile (1 tile, persistent, bf16)
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

    # CB 6: cb_mean - Group mean scalar tile (1 tile per group)
    cb_mean_desc = ttnn.CBDescriptor(
        total_size=1 * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    # CB 7: cb_den - Group rsqrt(var+eps) tile (1 tile per group)
    cb_den_desc = ttnn.CBDescriptor(
        total_size=1 * tile_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_DEN,
                data_format=input_tensor.dtype,
                page_size=tile_page_size,
            )
        ],
    )

    # CB 16: cb_normalized - Normalized output tiles (Ct per tile-row)
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

    # CB 17: cb_output_rm - Untilized RM data (Ct per tile-row)
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

    # CB 24: cb_sq_sum - E[x^2] accumulator (1 tile)
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

    # CB 25: cb_tmp - Scratch (1 tile)
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

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    # Compile-time args: stick_size, TensorAccessorArgs(input), TensorAccessorArgs(gamma), TensorAccessorArgs(beta)
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args())

    # Runtime args: input_addr, gamma_addr, beta_addr, num_sticks(=N*HW), Ct,
    #               block_width_size(=Ct*64), gamma_num_tiles(=Ct), beta_num_tiles(=Ct), packed_eps
    num_sticks = N * HW
    block_width_size = Ct * 64  # Each tile column occupies 64 bytes per row (32 elements * 2 bytes)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_tensor.buffer_address(),
        beta_tensor.buffer_address(),
        num_sticks,
        Ct,
        block_width_size,
        Ct,  # gamma_num_tiles
        Ct,  # beta_num_tiles
        eps_packed,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "group_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Compile-time args: Ht, Ct, G, Ct_g, N
    compute_ct_args = [Ht, Ct, G, Ct_g, N]

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
    # Compile-time args: cb_output_rm, output_stick_size, tile_height(=32),
    #                     num_tile_rows(=N*Ht), Ct, TensorAccessorArgs(output)
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
        ],
    )
