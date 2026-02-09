# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Row Standardize - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args for the
tilize -> row_standardize -> untilize pipeline.

Pipeline stages:
1. Reader: Read RM sticks from DRAM, generate scalar tiles (scaler, epsilon)
2. Compute: Tilize -> Mean -> Sub -> Square -> Var -> Add+Rsqrt -> Mul -> Untilize
3. Writer: Write RM sticks to DRAM

Work distribution: Single-core prototype (tile-rows processed sequentially)
"""

from pathlib import Path
import struct
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    epsilon: float,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for row_standardize operation.

    Args:
        input_tensor: Input tensor (on device, ROW_MAJOR layout)
        output_tensor: Pre-allocated output tensor (on device, ROW_MAJOR layout)
        epsilon: Small constant for numerical stability

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    dtype = input_tensor.dtype

    # Compute dimensions
    W = shape[-1]  # Last dimension (width)
    H = shape[-2]  # Second-to-last dimension (height)

    # Compute batch size (product of all dimensions except last two)
    N_batch = 1
    for i in range(len(shape) - 2):
        N_batch *= shape[i]

    # Tile dimensions
    Wt = W // 32  # Number of tiles per row
    Ht = H // 32  # Number of tiles per column
    nblocks = N_batch * Ht  # Total number of tile-rows to process

    # Compute page sizes and data formats
    # For ROW_MAJOR: page_size = W * element_size (one stick)
    element_size = input_tensor.element_size()  # 2 for bf16, 4 for f32
    stick_size_bytes = W * element_size

    # Tile size for circular buffers
    # CRITICAL: CB page size must match tile size for the data format
    tile_size = input_tensor.tile.get_tile_size(dtype)  # 2048 for bf16, 4096 for f32

    # Intermediate format: use float32 for intermediate CBs if input is float32
    # This matches the spec: when fp32_dest_acc_en is enabled, intermediates are f32
    is_float32 = dtype == ttnn.float32
    if is_float32:
        intermed_fmt = ttnn.float32
        intermed_tile_size = 4096
    else:
        intermed_fmt = ttnn.bfloat16
        intermed_tile_size = 2048

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Single-core prototype: all work on core (0, 0)
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. COMPUTE SCALER AND EPSILON VALUES ==========
    # Scaler = 1/W (for reduce operations)
    # Packed format for bfloat16: (bf16 << 16 | bf16)
    # Packed format for float32: reinterpreted float bits as uint32

    scaler_value = 1.0 / float(W)

    # CRITICAL: Reduce scaler is ALWAYS packed as bfloat16 (bf16 << 16 | bf16)
    # regardless of input dtype. This matches the softmax reference which always
    # uses Float16_b for the scaler CB. generate_reduce_scaler expects this format.
    scaler_bf16 = _float_to_bfloat16(scaler_value)
    scaler_packed = (scaler_bf16 << 16) | scaler_bf16

    if is_float32:
        # Float32 epsilon: reinterpret float bits as uint32
        epsilon_packed = struct.unpack("I", struct.pack("f", epsilon))[0]
    else:
        # Bfloat16 epsilon: pack as (bf16 << 16 | bf16)
        epsilon_bf16 = _float_to_bfloat16(epsilon)
        epsilon_packed = (epsilon_bf16 << 16) | epsilon_bf16

    # ========== 4. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices from spec:
    # c_0: cb_rm_in (RM sticks from reader)
    # c_1: cb_scaler (reduce scaler 1/W)
    # c_2: cb_eps (epsilon scalar)
    # c_3: cb_tilized (tilized input)
    # c_4: cb_tilized_out (normalized output tiles)
    # c_16: cb_rm_out (RM sticks to writer)
    # c_24: cb_mean (row means)
    # c_25: cb_xmm (x - mean)
    # c_26: cb_xmm_sq ((x-mean)^2)
    # c_27: cb_var (row variance)
    # c_28: cb_invstd (rsqrt(var + eps))

    cb_descriptors = []

    # cb_rm_in (c_0): Input RM sticks, capacity = Wt pages
    # Page size for ROW_MAJOR CB is still tile_size (spec says "Page size = tile_size for the dtype")
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_scaler (c_1): Reduce scaler tile (persistent), 1 tile
    # CRITICAL: Scaler CB is ALWAYS bfloat16, matching softmax reference pattern.
    # generate_reduce_scaler writes packed bf16 (bf16 << 16 | bf16) regardless of input dtype.
    bf16_tile_size = 2048  # 32*32*2 bytes for bfloat16
    cb_descriptors.append(
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

    # cb_eps (c_2): Epsilon scalar tile (persistent), 1 tile
    # Uses intermediate format (spec says epsilon added to intermediate variance)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=2,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_tilized (c_3): Tilized input tiles, capacity = Wt tiles
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=3,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_tilized_out (c_4): Normalized tiles, capacity = Wt tiles
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=4,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_rm_out (c_16): Output RM sticks, capacity = Wt pages
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=16,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_mean (c_24): Per-row mean (column vector), 1 tile
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=24,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_xmm (c_25): x - mean intermediate, capacity = Wt tiles
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=25,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_xmm_sq (c_26): (x-mean)^2 intermediate, capacity = Wt tiles
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=26,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_var (c_27): Per-row variance (column vector), 1 tile
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=27,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_invstd (c_28): rsqrt(var + eps), 1 tile
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=intermed_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=28,
                    data_format=intermed_fmt,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # ========== 5. KERNEL DESCRIPTORS ==========

    # ----- Reader Kernel -----
    # Compile-time args:
    #   0: stick_size_bytes
    #   1: is_float32
    #   2+: TensorAccessorArgs (src)
    reader_ct_args = [stick_size_bytes, 1 if is_float32 else 0]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Runtime args:
    #   0: src_addr
    #   1: num_sticks (total sticks to read = nblocks * 32)
    #   2: start_stick_id (0 for single-core)
    #   3: Wt
    #   4: scaler (packed)
    #   5: epsilon (packed)
    num_sticks = nblocks * 32
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_sticks,
        0,  # start_stick_id
        Wt,
        scaler_packed,
        epsilon_packed,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_standardize_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- Compute Kernel -----
    # Compile-time args:
    #   0: Wt
    #   1: nblocks
    compute_ct_args = [Wt, nblocks]

    # Runtime args: none (all work is determined by compile-time args)
    compute_rt_args = ttnn.RuntimeArgs()

    # Config: Use ComputeConfigDescriptor
    # CRITICAL: Enable fp32_dest_acc_en for float32 input
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=is_float32,
        math_approx_mode=False,
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_standardize_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    # ----- Writer Kernel -----
    # Compile-time args:
    #   0: stick_size_bytes
    #   1: Wt
    #   2+: TensorAccessorArgs (dst)
    writer_ct_args = [stick_size_bytes, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Runtime args:
    #   0: dst_addr
    #   1: num_blocks
    #   2: start_stick_id (0 for single-core)
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        nblocks,
        0,  # start_stick_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_standardize_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 6. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )


def _float_to_bfloat16(value: float) -> int:
    """
    Convert float32 to bfloat16 (16-bit representation).

    Bfloat16 is the upper 16 bits of float32 (sign + 8-bit exponent + 7-bit mantissa).
    Truncates the lower 16 bits of the mantissa.

    Returns:
        16-bit integer representing bfloat16 value
    """
    # Pack float as uint32
    bits = struct.unpack("I", struct.pack("f", value))[0]
    # Take upper 16 bits
    bf16 = (bits >> 16) & 0xFFFF
    return bf16
