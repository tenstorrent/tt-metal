# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
) -> ttnn.ProgramDescriptor:
    """
    Create a ProgramDescriptor for the softmax operation.

    CB layout:
        CB 0  (cb_input)      - Input tiles, R pages
        CB 8  (cb_scaler)     - Reduce scaler (1.0f), 1 page
        CB 16 (cb_out)        - Output tiles, 2 pages (double-buffered)
        CB 24 (cb_max)        - Max reduction result, 1 page
        CB 25 (cb_exp)        - exp(x - max) intermediate, R pages
        CB 26 (cb_recip_sum)  - 1/sum(exp) result, 1 page

    Where R = Wt for dim=-1, Ht for dim=-2.
    """
    # --- Tensor shape info ---
    shape = input_tensor.shape
    rank = len(shape)

    # Normalize to 4D
    if rank == 2:
        N, C, H, W = 1, 1, shape[0], shape[1]
    elif rank == 3:
        N, C, H, W = 1, shape[0], shape[1], shape[2]
    else:
        N, C, H, W = shape[0], shape[1], shape[2], shape[3]

    Ht = H // 32
    Wt = W // 32
    NC = N * C

    is_dim_h = 1 if dim == -2 else 0

    if dim == -1:
        R = Wt  # tiles per work unit (one row of tiles)
        total_work_units = NC * Ht  # one work unit per row
    else:  # dim == -2
        R = Ht  # tiles per work unit (one column of tiles)
        total_work_units = NC * Wt  # one work unit per column

    # --- Single-core execution for simplicity ---
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Page size (tile size for bfloat16 32x32) ---
    page_size = input_tensor.tile.get_tile_size(input_tensor.dtype)

    # --- CB indices ---
    CB_INPUT = 0
    CB_SCALER = 8
    CB_OUT = 16
    CB_MAX = 24
    CB_EXP = 25
    CB_RECIP_SUM = 26

    # --- Circular Buffer Descriptors ---
    # CB 0: input (R pages)
    cb_input_desc = ttnn.CBDescriptor(
        total_size=R * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # CB 8: scaler (1 page, bfloat16)
    cb_scaler_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=ttnn.bfloat16,
                page_size=page_size,
            )
        ],
    )

    # CB 16: output (2 pages, double-buffered)
    cb_out_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # CB 24: max (1 page)
    cb_max_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MAX,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # CB 25: exp intermediate (R pages)
    cb_exp_desc = ttnn.CBDescriptor(
        total_size=R * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXP,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # CB 26: recip_sum (1 page)
    cb_recip_sum_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RECIP_SUM,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    cbs = [
        cb_input_desc,
        cb_scaler_desc,
        cb_out_desc,
        cb_max_desc,
        cb_exp_desc,
        cb_recip_sum_desc,
    ]

    # --- Kernel compile-time args ---

    # Reader CT args: [cb_input, cb_scaler, R, is_dim_h, Wt] + TensorAccessorArgs(input)
    reader_ct_args = [CB_INPUT, CB_SCALER, R, is_dim_h, Wt]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Compute CT args: [cb_input, cb_scaler, cb_out, cb_max, cb_exp, cb_recip_sum, R, numeric_stable, num_work_units, is_dim_h]
    compute_ct_args = [
        CB_INPUT,
        CB_SCALER,
        CB_OUT,
        CB_MAX,
        CB_EXP,
        CB_RECIP_SUM,
        R,
        1 if numeric_stable else 0,
        total_work_units,
        is_dim_h,
    ]

    # Writer CT args: [cb_out, R, is_dim_h, Wt] + TensorAccessorArgs(output)
    writer_ct_args = [CB_OUT, R, is_dim_h, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # --- Runtime args ---
    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [src_addr, total_work_units, 0]

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [dst_addr, total_work_units, 0]

    # Compute has no runtime args (all info in CT args)
    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = []

    # --- Kernel path (relative to tt-metal repo root) ---
    kernel_base = "ttnn/ttnn/operations/softmax/kernels"

    # --- Kernel Descriptors ---
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{kernel_base}/softmax_reader.cpp",
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{kernel_base}/softmax_compute.cpp",
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{kernel_base}/softmax_writer.cpp",
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Program Descriptor ---
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        cbs=cbs,
        semaphores=[],
    )

    return program_descriptor
