# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for the layer normalization operation on row-major tensors.
"""

from pathlib import Path
import struct
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    epsilon: float,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm operation.

    Args:
        input_tensor: Input tensor (on device, ROW_MAJOR layout)
        gamma: Scale parameter (on device, ROW_MAJOR layout)
        beta: Shift parameter (on device, ROW_MAJOR layout)
        output_tensor: Pre-allocated output tensor (on device)
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    dtype = input_tensor.dtype
    element_size = input_tensor.element_size()

    # Extract dimensions
    input_shape = input_tensor.shape
    H = input_shape[-2]
    W = input_shape[-1]

    # Calculate tile dimensions
    Wt = W // 32  # Number of tiles along width
    Ht = H // 32  # Number of tiles along height

    # Calculate total number of rows/sticks (product of all dims except the last)
    # For shape (N, C, H, W): num_sticks = N * C * H
    num_sticks = 1
    for i in range(len(input_shape) - 1):
        num_sticks *= input_shape[i]

    # Number of tile-rows to process
    num_tile_rows = num_sticks // 32

    # Calculate stick size (one row of W elements)
    stick_size = W * element_size

    # Calculate tile size for the given dtype
    tile_size = ttnn.tile_size(dtype)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Single core execution (per spec)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB assignments from spec:
    # c_0:  cb_input_rm (input RM sticks staging)
    # c_1:  cb_reduce_scaler (1/W scaler tile)
    # c_2:  cb_input_tilized (tilized input)
    # c_3:  cb_gamma_rm (gamma RM sticks staging)
    # c_4:  cb_beta_rm (beta RM sticks staging)
    # c_5:  cb_gamma_tilized (tilized gamma, persistent)
    # c_6:  cb_beta_tilized (tilized beta, persistent)
    # c_7:  cb_eps_scalar (epsilon scalar tile)
    # c_16: cb_output_rm (output RM sticks)
    # c_24: cb_mean (row-wise mean, 1 tile)
    # c_25: cb_centered (x - mean, Wt tiles)
    # c_26: cb_squared ((x - mean)^2, Wt tiles)
    # c_27: cb_var (variance + epsilon, 1 tile)
    # c_28: cb_rstd (1/sqrt(var+eps), 1 tile)
    # c_29: cb_normalized ((x-mean)*rstd, Wt tiles)
    # c_30: cb_gamma_applied (gamma * normalized, Wt tiles)
    # c_31: cb_out_tilized (final tilized output, Wt tiles)

    cbs = []

    # Input/output data format (bfloat16 or float32)
    input_data_format = dtype

    # Intermediate data format (use float32 for better precision if available)
    # For now, keep same as input for simplicity
    intermediate_data_format = dtype
    intermediate_tile_size = tile_size

    # c_0: Input RM sticks (Wt tiles capacity)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: Reduce scaler tile (1 tile) - ALWAYS bfloat16
    # The reduce hardware scaler is always in bfloat16 format regardless of input dtype.
    bf16_tile_size = ttnn.tile_size(ttnn.bfloat16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=1,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_2: Input tilized (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=2,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_3: Gamma RM sticks (Wt tiles capacity)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=3,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_4: Beta RM sticks (Wt tiles capacity)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=4,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_5: Gamma tilized (Wt tiles, persistent)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=5,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_6: Beta tilized (Wt tiles, persistent)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=6,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_7: Epsilon scalar tile (1 tile)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=7,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_16: Output RM sticks (Wt tiles capacity)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=16,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_24: Mean (1 tile)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=24,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_25: Centered (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=25,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_26: Squared (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=26,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_27: Variance (1 tile)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=27,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_28: Rstd (1 tile)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=28,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_29: Normalized (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=29,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_30: Gamma applied (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermediate_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=30,
                    data_format=intermediate_data_format,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # c_31: Output tilized (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=31,
                    data_format=input_data_format,
                    page_size=tile_size,
                )
            ],
        )
    )

    # ========== 4. PREPARE SCALAR VALUES ==========

    # Helper: pack float as bfloat16 double-packed: (bf16 << 16 | bf16)
    def pack_bf16(value):
        float_bits = struct.unpack(">I", struct.pack(">f", value))[0]
        bf16_bits = (float_bits >> 16) & 0xFFFF
        return (bf16_bits << 16) | bf16_bits

    # Helper: pack float as raw float32 bits
    def pack_f32(value):
        return struct.unpack(">I", struct.pack(">f", value))[0]

    # Reduce scaler (1/W): ALWAYS bfloat16 packed (bf16 << 16 | bf16)
    # The reduce hardware scaler tile is always in bfloat16 format,
    # regardless of the input data format.
    scaler_value = 1.0 / float(W)
    reduce_scaler = pack_bf16(scaler_value)

    # Epsilon scalar: depends on the data format
    # For bfloat16: (bf16 << 16 | bf16)
    # For float32: raw float32 bits
    if dtype == ttnn.bfloat16:
        eps_scalar = pack_bf16(epsilon)
    else:  # float32
        eps_scalar = pack_f32(epsilon)

    # ========== 5. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = []
    reader_ct_args.append(stick_size)  # stick_size
    reader_ct_args.append(1 if dtype == ttnn.float32 else 0)  # is_float32
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    # Gamma/beta parameters for reading
    # Gamma and beta are shape [..., 1, W] - we need to read the W-element row and replicate 32 times
    gamma_num_sticks = 1  # We'll read just 1 row and replicate
    gamma_stick_size = stick_size

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address(),
        beta.buffer_address(),
        num_sticks,
        Wt,
        Wt * 32 * element_size,  # block_width_size (bytes per tile-row)
        reduce_scaler,
        eps_scalar,
        gamma_num_sticks,
        gamma_stick_size,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.RISCV_0_default,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [16]  # cb_id_out
    writer_ct_args.append(stick_size)  # output_stick_size
    writer_ct_args.append(32)  # tile_height
    writer_ct_args.append(num_tile_rows)  # num_blocks_across_height
    writer_ct_args.append(1)  # num_blocks_per_row (WSmall pattern)
    writer_ct_args.append(Wt)  # num_tiles_per_block
    writer_ct_args.append(Wt * 32 * element_size)  # block_width_bytes
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1,
            noc=ttnn.NOC.RISCV_1_default,
        ),
    )

    # --- Compute kernel ---
    compute_ct_args = [
        num_tile_rows,
        Wt,
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 6. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
