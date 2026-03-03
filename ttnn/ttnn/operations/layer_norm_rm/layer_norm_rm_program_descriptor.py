# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Configures circular buffers, work distribution, and kernel descriptors for
the row-major layer normalization operation.

Architecture:
  - Reader: reads RM sticks for input (per tile-row), gamma, beta (once per core)
  - Compute: tilize -> layer norm (mean, center, var, rsqrt, affine) -> untilize
  - Writer: writes RM sticks for output
"""

import struct
from pathlib import Path
import ttnn

# Kernel files are collocated in the kernels/ subdirectory within the operation
KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices matching the design document
CB_IN_RM = 0  # Input RM sticks from reader
CB_TILIZED = 1  # Tilized input tiles (compute intermediate)
CB_SCALER = 2  # Reduce scaler tile (1/W)
CB_EPS = 3  # Epsilon scaler tile
CB_GAMMA = 4  # Tilized gamma (persistent)
CB_BETA = 5  # Tilized beta (persistent)
CB_GAMMA_RM = 6  # RM gamma sticks (32 repeated, staging for tilize)
CB_BETA_RM = 7  # RM beta sticks (32 repeated, staging for tilize)
CB_OUT = 16  # Output RM sticks for writer
CB_MEAN = 24  # Row-wise mean (1 tile)
CB_CENTERED = 25  # x - mean (Wt tiles)
CB_INV_STD = 26  # rsqrt(var + eps) scratch (1 tile)
CB_NORMED = 27  # General Wt-tile scratch (squared, then normed)
CB_AFFINE = 28  # General Wt-tile scratch (gamma*normed, result before +beta)


def _float_to_bfloat16_packed(f: float) -> int:
    """
    Pack a float value into the format required by generate_reduce_scaler.

    The reduce scaler is stored as a bfloat16 value packed into a uint32:
    (bf16_bits << 16 | bf16_bits)
    """
    # Pack float to 4 bytes, take the upper 2 bytes as bfloat16
    packed = struct.pack(">f", f)
    bf16_bits = (packed[0] << 8) | packed[1]
    return (bf16_bits << 16) | bf16_bits


def _float_to_uint32(f: float) -> int:
    """Convert a float to its IEEE 754 bit representation as uint32."""
    return struct.unpack("I", struct.pack("f", f))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor,
    beta_tensor,
    output_tensor: ttnn.Tensor,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor:  Input tensor (ROW_MAJOR, bfloat16, on device)
        gamma_tensor:  Gamma tensor or None
        beta_tensor:   Beta tensor or None
        output_tensor: Pre-allocated output tensor (ROW_MAJOR, bfloat16, on device)
        epsilon:       Numerical stability constant

    Returns:
        ProgramDescriptor for use with ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)

    # Last two dims are H (tile rows * 32) and W
    W = shape[-1]
    H = shape[-2]

    # Compute total number of outer (batch) elements
    outer_volume = 1
    for i in range(rank - 2):
        outer_volume *= shape[i]

    # Number of "tile-rows" (each tile-row = 32 rows = 1 tile height)
    Ht = H // 32  # number of tile rows in H dimension
    Wt = W // 32  # number of tiles per row in W dimension
    total_tile_rows = outer_volume * Ht  # total tile-rows across all batches

    # RM stick size: one row of width W elements, each bfloat16 (2 bytes)
    # For ROW_MAJOR tensors, buffer_page_size() == stick_size (W * bytes_per_element)
    stick_size_bytes = input_tensor.buffer_page_size()

    # Tile size for bfloat16 (32*32*2 = 2048 bytes)
    tile_size_bytes = ttnn.tile_size(input_tensor.dtype)

    # ========== 2. WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tile_rows_per_core_group_1,
        tile_rows_per_core_group_2,
    ) = ttnn.split_work_to_cores(compute_grid_size, total_tile_rows)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========

    # CB 0: Input RM sticks (batched as Wt tile-sized pages)
    # Each page holds data for one tile column of one RM stick.
    # Reader batches 32 sticks into Wt tile-pages for the tilize pattern.
    cb_in_rm_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_IN_RM,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 1: Tilized input tiles (Wt tiles)
    cb_tilized_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TILIZED,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 2: Reduce scaler (1 tile, program-lifetime)
    cb_scaler_descriptor = ttnn.CBDescriptor(
        total_size=tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 3: Epsilon tile (1 tile, program-lifetime)
    cb_eps_descriptor = ttnn.CBDescriptor(
        total_size=tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EPS,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 4: Tilized gamma (Wt tiles, program-lifetime, persistent)
    cb_gamma_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 5: Tilized beta (Wt tiles, program-lifetime, persistent)
    cb_beta_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 6: RM gamma sticks staging (32 repeated copies => Wt tile-pages)
    cb_gamma_rm_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA_RM,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 7: RM beta sticks staging (32 repeated copies => Wt tile-pages)
    cb_beta_rm_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA_RM,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 16: Output RM sticks (Wt tile-pages)
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 24: Row-wise mean (1 tile)
    cb_mean_descriptor = ttnn.CBDescriptor(
        total_size=tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 25: Centered values x - mean (Wt tiles)
    cb_centered_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_CENTERED,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 26: inv_std scratch (1 tile: variance -> rsqrt(var+eps))
    cb_inv_std_descriptor = ttnn.CBDescriptor(
        total_size=tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INV_STD,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 27: General Wt-tile scratch (squared values, then normed output)
    cb_normed_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_NORMED,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    # CB 28: General Wt-tile scratch (gamma*normed, result before +beta)
    cb_affine_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_AFFINE,
                data_format=ttnn.bfloat16,
                page_size=tile_size_bytes,
            )
        ],
    )

    all_cbs = [
        cb_in_rm_descriptor,
        cb_tilized_descriptor,
        cb_scaler_descriptor,
        cb_eps_descriptor,
        cb_gamma_descriptor,
        cb_beta_descriptor,
        cb_gamma_rm_descriptor,
        cb_beta_rm_descriptor,
        cb_out_descriptor,
        cb_mean_descriptor,
        cb_centered_descriptor,
        cb_inv_std_descriptor,
        cb_normed_descriptor,
        cb_affine_descriptor,
    ]

    # ========== 4. COMPILE-TIME ARGS ==========

    # Reader compile-time args:
    #   [0] stick_size (bytes): one RM stick = W * element_size
    #   [1] gamma_stick_size (bytes): same (gamma has same W)
    #   [2+] TensorAccessorArgs for input
    #   [N+] TensorAccessorArgs for gamma (if present)
    #   [M+] TensorAccessorArgs for beta (if present)
    has_gamma = gamma_tensor is not None
    has_beta = beta_tensor is not None

    reader_ct_args = [
        stick_size_bytes,
        stick_size_bytes,  # gamma/beta stick size (same W)
        int(has_gamma),  # flag: 1 if gamma present
        int(has_beta),  # flag: 1 if beta present
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args())
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args())

    # Writer compile-time args:
    #   [0] stick_size (bytes)
    #   [1+] TensorAccessorArgs for output
    writer_ct_args = [stick_size_bytes]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute compile-time args:
    #   [0] Wt: tiles per row
    #   [1] has_gamma: 1 if gamma present
    #   [2] has_beta: 1 if beta present
    compute_ct_args = [
        Wt,
        int(has_gamma),
        int(has_beta),
    ]

    # ========== 5. RUNTIME ARGS ==========
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    # Scaler for reduce: 1/W packed as bfloat16
    scaler_value = _float_to_bfloat16_packed(1.0 / W)
    # Epsilon as bfloat16 packed (for epsilon tile generation)
    eps_value = _float_to_bfloat16_packed(epsilon)

    # Set runtime args for all cores in the compute grid
    grid_width = compute_grid_size.x
    grid_height = compute_grid_size.y

    # Build sets of active cores and their tile-row assignments
    active_core_tile_rows = {}  # (x, y) -> num_tile_rows_this_core

    start_tile_row = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                active_core_tile_rows[(x, y)] = (tile_rows_per_core_group_1, start_tile_row)
                start_tile_row += tile_rows_per_core_group_1

    if tile_rows_per_core_group_2 > 0:
        for core_range in core_group_2.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    active_core_tile_rows[(x, y)] = (tile_rows_per_core_group_2, start_tile_row)
                    start_tile_row += tile_rows_per_core_group_2

    # Set runtime args for all cores in the full grid
    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) in active_core_tile_rows:
                num_tile_rows_this_core, start_tr = active_core_tile_rows[(x, y)]
                # Number of actual RM sticks this core processes
                num_sticks = num_tile_rows_this_core * 32
                # Stick ID offset (each tile-row = 32 sticks)
                start_stick_id = start_tr * 32

                # Reader runtime args:
                #   [0] src_addr: input buffer base address
                #   [1] gamma_addr: gamma buffer address (0 if no gamma)
                #   [2] beta_addr: beta buffer address (0 if no beta)
                #   [3] num_sticks: total sticks this core reads
                #   [4] start_stick_id: first stick for this core
                #   [5] scaler_value: 1/W as bf16-packed uint32
                #   [6] eps_value: epsilon as bf16-packed uint32
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),
                    gamma_tensor.buffer_address() if has_gamma else 0,
                    beta_tensor.buffer_address() if has_beta else 0,
                    num_sticks,
                    start_stick_id,
                    scaler_value,
                    eps_value,
                ]

                # Writer runtime args:
                #   [0] dst_addr: output buffer base address
                #   [1] num_sticks: total sticks this core writes
                #   [2] start_stick_id: first output stick for this core
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    num_sticks,
                    start_stick_id,
                ]

                # Compute runtime args:
                #   [0] num_rows_per_core: tile-rows this core processes
                compute_rt_args[x][y] = [
                    num_tile_rows_this_core,
                ]
            else:
                # MUST set empty args for idle cores
                reader_rt_args[x][y] = []
                writer_rt_args[x][y] = []
                compute_rt_args[x][y] = []

    # ========== 6. KERNEL DESCRIPTORS ==========

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 7. ASSEMBLE PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
