# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import struct


def _bf16_to_uint32(value: float) -> int:
    """Convert a float to bfloat16 and return as uint32 for packing."""
    # Convert to float32 bytes
    f32_bytes = struct.pack("f", value)
    f32_int = struct.unpack("I", f32_bytes)[0]
    # Extract upper 16 bits (bfloat16)
    bf16 = (f32_int >> 16) & 0xFFFF
    return bf16


def _pack_bf16_pair(value: float) -> int:
    """Pack a bfloat16 value as (bf16 << 16 | bf16) for reduce scaler."""
    bf16 = _bf16_to_uint32(value)
    return (bf16 << 16) | bf16


def _pack_scalar_for_dtype(value: float, dtype) -> int:
    """Pack a scalar value according to dtype requirements."""
    if dtype == ttnn.bfloat16:
        # For bfloat16, pack as (bf16 << 16 | bf16)
        return _pack_bf16_pair(value)
    else:  # float32
        # For float32, bit-cast the float32 value
        return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor, gamma: ttnn.Tensor, beta: ttnn.Tensor, output_tensor: ttnn.Tensor, epsilon: float
):
    """
    Create program descriptor for layer_norm_rm operation.

    Args:
        input_tensor: Input tensor in ROW_MAJOR layout
        gamma: Gamma tensor (scale parameter), or None
        beta: Beta tensor (bias parameter), or None
        output_tensor: Pre-allocated output tensor
        epsilon: Small constant for numerical stability

    Returns:
        ProgramDescriptor for ttnn.generic_op
    """

    # Extract tensor properties
    input_shape = input_tensor.shape
    dtype = input_tensor.dtype
    element_size = input_tensor.element_size()

    # Calculate dimensions
    W = input_shape[-1]
    H = input_shape[-2]

    # Calculate total number of sticks (flattened H dimension)
    num_sticks = 1
    for i in range(len(input_shape) - 1):
        num_sticks *= input_shape[i]

    Wt = W // 32  # Number of tiles along W
    num_tile_rows = num_sticks // 32  # Number of tile-rows

    # Stick sizes
    stick_size = W * element_size

    # Calculate reduce scaler (1/W)
    reduce_scaler_value = 1.0 / W
    reduce_scaler_packed = _pack_bf16_pair(reduce_scaler_value)  # Always bfloat16 for reduce

    # Calculate epsilon scalar - pack according to input dtype
    eps_scalar_packed = _pack_scalar_for_dtype(epsilon, dtype)

    # Determine intermediate dtype
    # Use same format as input for CB storage to avoid unpacker reconfig issues.
    # Precision is maintained via fp32_dest_acc_en=True (FPU accumulates in float32).
    intermed_dtype = dtype  # Match input format (bfloat16 or float32)
    fp32_dest_acc_en = True

    # Get device
    device = input_tensor.device()

    # Single-core execution
    compute_grid_size = device.compute_with_storage_grid_size()
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    core_range_set = ttnn.CoreRangeSet({core_range})

    # Get tile sizes for page size calculation
    input_tile = input_tensor.tile
    input_page_size = input_tile.get_tile_size(dtype)

    intermed_page_size = input_tile.get_tile_size(intermed_dtype)

    # CB data types - CBFormatDescriptor accepts DataType directly
    input_data_format = dtype  # bfloat16 or float32
    intermed_data_format = intermed_dtype  # Same as input format

    # --- Circular Buffer Configuration ---

    cbs = []

    # CB 0: cb_input_rm - Input row-major sticks from reader
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 1: cb_tilized_input - Tilized input tiles
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 2: cb_gamma_rm - Gamma row-major sticks from reader
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 3: cb_gamma_tilized - Gamma tilized tiles (persistent)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=3, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 4: cb_beta_rm - Beta row-major sticks from reader
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=4, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 5: cb_beta_tilized - Beta tilized tiles (persistent)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=5, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 6: cb_reduce_scaler - Reduce scaler tile (ALWAYS bfloat16)
    bf16_page_size = input_tile.get_tile_size(ttnn.bfloat16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=bf16_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=6,
                    data_format=ttnn.bfloat16,  # ALWAYS bfloat16
                    page_size=bf16_page_size,
                )
            ],
        )
    )

    # CB 7: cb_eps_scalar - Epsilon scalar tile (matches intermediate format)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=7, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 8: cb_output_tiles - Final output tiles before untilize
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=8, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 16: cb_output_rm - Output row-major sticks for writer
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=16, data_format=input_data_format, page_size=input_page_size)
            ],
        )
    )

    # CB 24: cb_mean - Row-wise mean (1 tile, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=24, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 25: cb_centered - x - mean intermediate (Wt tiles, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=25, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 26: cb_centered_sq - (x - mean)^2 intermediate (Wt tiles, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=26, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 27: cb_var - Row-wise variance (1 tile, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=27, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 28: cb_rstd - 1/sqrt(var+eps) (1 tile, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=28, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 29: cb_normed - x_centered * rstd (Wt tiles, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=29, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # CB 30: cb_gamma_applied - gamma * normed (Wt tiles, intermediate precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_page_size,
            core_ranges=core_range_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=30, data_format=intermed_data_format, page_size=intermed_page_size)
            ],
        )
    )

    # --- Kernel Configuration ---

    # Reader kernel compile-time args
    reader_compile_args = [
        stick_size,  # 0: stick_size
        stick_size,  # 1: gamma_beta_stick_size (same as input stick size)
    ]
    # Add TensorAccessor args for input
    reader_compile_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Add TensorAccessor args for gamma (if provided, else dummy)
    if gamma is not None:
        reader_compile_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        # Non-sharded dummy accessor: exactly 1 compile-time arg (args_config = 0)
        reader_compile_args.extend([0])
    # Add TensorAccessor args for beta (if provided, else dummy)
    if beta is not None:
        reader_compile_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    else:
        reader_compile_args.extend([0])

    # Reader kernel runtime args
    reader_rt_args = ttnn.RuntimeArgs()
    input_buffer_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    reader_rt_args[0][0] = [
        input_buffer_addr,  # 0: src_addr
        gamma_addr,  # 1: gamma_addr
        beta_addr,  # 2: beta_addr
        num_sticks,  # 3: num_sticks
        num_tile_rows,  # 4: num_tile_rows
        Wt,  # 5: Wt
        reduce_scaler_packed,  # 6: reduce_scaler (packed bfloat16)
        eps_scalar_packed,  # 7: eps_scalar (packed according to dtype)
    ]

    # Reader kernel descriptor
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/layer_norm_rm/kernels/layer_norm_rm_reader.cpp",
        core_ranges=core_range_set,
        compile_time_args=reader_compile_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Compute kernel compile-time args
    compute_compile_args = [
        Wt,  # 0: Wt
        num_tile_rows,  # 1: num_tile_rows
        1 if gamma is not None else 0,  # 2: has_gamma
        1 if beta is not None else 0,  # 3: has_beta
    ]

    # Compute kernel runtime args (none for this operation)
    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[0][0] = []

    # Compute kernel descriptor
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/layer_norm_rm/kernels/layer_norm_rm_compute.cpp",
        core_ranges=core_range_set,
        compile_time_args=compute_compile_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=False,
        ),
    )

    # Writer kernel compile-time args
    writer_compile_args = [
        stick_size,  # 0: output_stick_size
        32,  # 1: tile_height
        num_tile_rows,  # 2: num_tile_rows
        Wt,  # 3: Wt
    ]
    # Add TensorAccessor args for output
    writer_compile_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Writer kernel runtime args
    writer_rt_args = ttnn.RuntimeArgs()
    output_buffer_addr = output_tensor.buffer_address()

    writer_rt_args[0][0] = [
        output_buffer_addr,  # 0: dst_addr
    ]

    # Writer kernel descriptor
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/layer_norm_rm/kernels/layer_norm_rm_writer.cpp",
        core_ranges=core_range_set,
        compile_time_args=writer_compile_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Assemble Program Descriptor ---

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel], cbs=cbs, semaphores=[]
    )

    return program_descriptor
