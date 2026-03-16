# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for the softmax operation."""

import os

import ttnn


# Kernel paths relative to tt-metal repo root
_KERNEL_DIR = os.path.join("ttnn", "ttnn", "operations", "softmax", "kernels")
_READER_KERNEL = os.path.join(_KERNEL_DIR, "reader_softmax.cpp")
_COMPUTE_KERNEL = os.path.join(_KERNEL_DIR, "compute_softmax.cpp")
_WRITER_KERNEL = os.path.join(_KERNEL_DIR, "writer_softmax.cpp")

# CB indices (following convention: 0-7 inputs, 8-15 special, 16-23 outputs, 24-31 intermediates)
CB_INPUT = 0  # Input tiles from reader
CB_SCALER = 1  # Reduce scaler tile (all 1.0 for MAX, SUM)
CB_OUTPUT = 16  # Output tiles to writer
CB_MAX = 24  # Reduced max tile (per row/col)
CB_EXP = 25  # Exp intermediate (sub + exp output)
CB_RECIP = 26  # 1/sum(exp) tile


def create_softmax_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
) -> ttnn.ProgramDescriptor:
    """
    Create a ProgramDescriptor for the softmax operation.

    Args:
        input_tensor: Input tensor (bfloat16, TILE_LAYOUT).
        output_tensor: Pre-allocated output tensor (same shape/dtype/layout).
        dim: Reduction dimension (-1 for width, -2 for height).
        numeric_stable: Whether to subtract max before exp.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op().
    """
    device = input_tensor.device()
    shape = input_tensor.shape

    # Compute tile dimensions
    # For rank >= 2 tensors, the last two dims are H and W
    rank = len(shape)
    H = shape[rank - 2]
    W = shape[rank - 1]
    Ht = H // 32  # tiles in height
    Wt = W // 32  # tiles in width
    HtWt = Ht * Wt  # tiles per batch slice

    # NC = product of all dims except H and W (batch dimensions)
    NC = 1
    for i in range(rank - 2):
        NC *= shape[i]

    # dim flag: 0 = width (dim=-1), 1 = height (dim=-2)
    dim_flag = 0 if dim == -1 else 1

    # numeric_stable flag: 1 = stable, 0 = unstable
    stable_flag = 1 if numeric_stable else 0

    # --- Work Distribution ---
    # dim=-1: work unit = tile-row -> total_units = NC * Ht
    # dim=-2: work unit = tile-column -> total_units = NC * Wt
    if dim == -1:
        total_work_units = NC * Ht
    else:
        total_work_units = NC * Wt

    compute_grid_size = device.compute_with_storage_grid_size()
    max_x = compute_grid_size.x - 1
    max_y = compute_grid_size.y - 1
    all_cores_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_x, max_y))])

    (
        num_cores,
        core_grid,
        core_group_1,
        core_group_2,
        work_per_core_group_1,
        work_per_core_group_2,
    ) = ttnn.split_work_to_cores(all_cores_range, total_work_units)

    # --- Circular Buffer Configuration ---
    page_size = input_tensor.tile.get_tile_size(input_tensor.dtype)

    # CB 0: input (double-buffered, 2 pages)
    cb_input_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT,
                data_format=ttnn.bfloat16,
                page_size=page_size,
            )
        ],
    )

    # CB 1: scaler (single page, persistent)
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

    # CB 16: output (double-buffered, 2 pages)
    cb_output_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT,
                data_format=ttnn.bfloat16,
                page_size=page_size,
            )
        ],
    )

    # CB 24: max tile (single page)
    cb_max_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MAX,
                data_format=ttnn.bfloat16,
                page_size=page_size,
            )
        ],
    )

    # CB 25: exp intermediate (double-buffered, 2 pages)
    cb_exp_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXP,
                data_format=ttnn.bfloat16,
                page_size=page_size,
            )
        ],
    )

    # CB 26: recip tile (single page)
    cb_recip_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RECIP,
                data_format=ttnn.bfloat16,
                page_size=page_size,
            )
        ],
    )

    all_cbs = [
        cb_input_desc,
        cb_scaler_desc,
        cb_output_desc,
        cb_max_desc,
        cb_exp_desc,
        cb_recip_desc,
    ]

    # --- Compile-time args ---

    # Reader CT args: Ht, Wt, HtWt, dim, numeric_stable, TensorAccessorArgs(input)
    reader_ct_args = [Ht, Wt, HtWt, dim_flag, stable_flag]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Writer CT args: Ht, Wt, HtWt, dim, TensorAccessorArgs(output)
    writer_ct_args = [Ht, Wt, HtWt, dim_flag]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute CT args: Ht, Wt, num_rows_or_cols (per-core), dim, numeric_stable
    # Note: num_rows_or_cols is per-core, so we use a placeholder here.
    # We'll create separate kernel descriptors for core_group_1 and core_group_2
    # if they have different work counts. Since compile-time args are per-kernel
    # and kernel is per-core-range, we need two compute kernels if groups differ.
    # For simplicity and correctness, we pass work_per_core as a runtime arg instead
    # of compile-time arg, OR we create two kernel descriptors.
    #
    # However, the design says compute has no runtime args and num_rows_or_cols
    # is a compile-time arg. We need separate core groups then.

    # --- Runtime args ---
    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    # Build runtime args for reader and writer
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    # Track start work unit for each core
    current_work_unit = 0

    # Set runtime args for core_group_1
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [src_addr, work_per_core_group_1, current_work_unit]
                writer_rt_args[x][y] = [dst_addr, work_per_core_group_1, current_work_unit]
                current_work_unit += work_per_core_group_1

    # Set runtime args for core_group_2 (if any)
    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [src_addr, work_per_core_group_2, current_work_unit]
                writer_rt_args[x][y] = [dst_addr, work_per_core_group_2, current_work_unit]
                current_work_unit += work_per_core_group_2

    # Set empty runtime args for all unused cores in the grid
    for x in range(compute_grid_size.x):
        for y in range(compute_grid_size.y):
            # Check if this core already has runtime args set
            # We do this by checking if it's in either core group
            core_in_group = False
            for core_range in core_grid.ranges():
                if core_range.start.x <= x <= core_range.end.x and core_range.start.y <= y <= core_range.end.y:
                    core_in_group = True
                    break
            if not core_in_group:
                reader_rt_args[x][y] = []
                writer_rt_args[x][y] = []

    # --- Kernel descriptors ---
    # For the compute kernel, num_rows_or_cols is a compile-time arg and differs
    # between core_group_1 and core_group_2. We handle this by creating
    # separate kernel descriptors for each group.

    kernels = []

    # Reader kernel - same CT args for all cores, per-core RT args differ
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    kernels.append(reader_kernel)

    # Compute kernel(s) - need separate descriptors per core group for different CT args
    compute_ct_args_g1 = [Ht, Wt, work_per_core_group_1, dim_flag, stable_flag]

    has_group_2 = work_per_core_group_2 > 0 and len(core_group_2.ranges()) > 0

    if not has_group_2:
        # Single group - all cores have the same work count
        compute_kernel = ttnn.KernelDescriptor(
            kernel_source=_COMPUTE_KERNEL,
            core_ranges=core_grid,
            compile_time_args=compute_ct_args_g1,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            ),
        )
        kernels.append(compute_kernel)
    else:
        # Two groups with different work counts
        compute_kernel_g1 = ttnn.KernelDescriptor(
            kernel_source=_COMPUTE_KERNEL,
            core_ranges=core_group_1,
            compile_time_args=compute_ct_args_g1,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            ),
        )
        kernels.append(compute_kernel_g1)

        compute_ct_args_g2 = [Ht, Wt, work_per_core_group_2, dim_flag, stable_flag]
        compute_kernel_g2 = ttnn.KernelDescriptor(
            kernel_source=_COMPUTE_KERNEL,
            core_ranges=core_group_2,
            compile_time_args=compute_ct_args_g2,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            ),
        )
        kernels.append(compute_kernel_g2)

    # Writer kernel - same CT args for all cores, per-core RT args differ
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    kernels.append(writer_kernel)

    # --- Assemble ProgramDescriptor ---
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=kernels,
        cbs=all_cbs,
        semaphores=[],
    )

    return program_descriptor
