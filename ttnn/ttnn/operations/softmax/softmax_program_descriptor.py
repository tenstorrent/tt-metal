# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for the softmax operation.

Work distribution:
  - dim=-1 (W): work unit = one tile-row (Wt tiles). Total units = NC * Ht.
  - dim=-2 (H): work unit = one tile-column (Ht tiles). Total units = NC * Wt.

Circular buffers:
  - c_0  (cb_input):      Input tiles from reader (max(Ht, Wt) pages)
  - c_1  (cb_scaler):     Reduce scaler 1.0 (1 page, bfloat16)
  - c_16 (cb_out):        Output tiles to writer (2 pages, double-buffered)
  - c_24 (cb_max):        Max reduction output (max(Ht, Wt) pages)
  - c_25 (cb_exps):       exp(x - max) intermediate (max(Ht, Wt) pages)
  - c_26 (cb_recip_sum):  1/sum(exp) output (max(Ht, Wt) pages)
"""

from pathlib import Path
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the softmax operation.

    Args:
        input_tensor: Input tensor (on device, bfloat16, TILE_LAYOUT, 4D)
        output_tensor: Pre-allocated output tensor (on device)
        dim: Reduction dimension (-1 for W, -2 for H)
        numeric_stable: Whether to subtract max before exp

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)

    # Extract dimensions: handle 2D, 3D, 4D by treating first dims as batch
    H = shape[rank - 2]
    W = shape[rank - 1]

    # Compute batch count NC = product of all dims except H, W
    NC = 1
    for i in range(rank - 2):
        NC *= shape[i]

    # Tile dimensions (always 32x32 for TILE_LAYOUT)
    TILE_H = 32
    TILE_W = 32
    Ht = H // TILE_H
    Wt = W // TILE_W
    HtWt = Ht * Wt

    # Page size from tensor metadata
    page_size = input_tensor.buffer_page_size()

    # Dimension flags
    is_dim_w = 1 if dim == -1 else 0

    # For dim=-1: work unit = tile-row, total = NC * Ht
    # For dim=-2: work unit = tile-column, total = NC * Wt
    if dim == -1:
        total_work_units = NC * Ht
        slice_tiles = Wt  # tiles per slice (row)
    else:
        total_work_units = NC * Wt
        slice_tiles = Ht  # tiles per slice (column)

    # CB page counts: unified sizing uses max(Ht, Wt)
    max_ht_wt = max(Ht, Wt)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        num_slices_per_core_group_1,
        num_slices_per_core_group_2,
    ) = ttnn.split_work_to_cores(compute_grid_size, total_work_units)

    grid_width = compute_grid_size.x
    grid_height = compute_grid_size.y

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices
    CB_INPUT = 0
    CB_SCALER = 1
    CB_OUT = 16
    CB_MAX = 24
    CB_EXPS = 25
    CB_RECIP_SUM = 26

    # cb_input: holds full slice (max(Ht, Wt) pages) for WaitUpfrontNoPop
    cb_input_descriptor = ttnn.CBDescriptor(
        total_size=max_ht_wt * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # cb_scaler: 1 page, bfloat16, persistent reduce scaler (value = 1.0)
    # Scaler must be bfloat16 regardless of input dtype
    scaler_page_size = page_size  # same tile size since input is already bfloat16
    cb_scaler_descriptor = ttnn.CBDescriptor(
        total_size=1 * scaler_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=ttnn.bfloat16,
                page_size=scaler_page_size,
            )
        ],
    )

    # cb_out: 2 pages, double-buffered for compute/writer overlap
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT,
                data_format=output_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # cb_max: max reduction output (max(Ht, Wt) pages)
    cb_max_descriptor = ttnn.CBDescriptor(
        total_size=max_ht_wt * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MAX,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # cb_exps: exp(x - max) intermediate (max(Ht, Wt) pages)
    cb_exps_descriptor = ttnn.CBDescriptor(
        total_size=max_ht_wt * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXPS,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # cb_recip_sum: 1/sum(exp) output (max(Ht, Wt) pages)
    cb_recip_sum_descriptor = ttnn.CBDescriptor(
        total_size=max_ht_wt * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RECIP_SUM,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    # CT args: Ht, Wt, HtWt, is_dim_w, TensorAccessorArgs(input)
    reader_ct_args = [Ht, Wt, HtWt, is_dim_w]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    # Build per-core runtime args
    slice_offset_group1 = 0
    for core_range in core_group_1.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),
                    num_slices_per_core_group_1,
                    slice_offset_group1,
                ]
                slice_offset_group1 += num_slices_per_core_group_1

    if core_group_2.num_cores() > 0:
        slice_offset_group2 = slice_offset_group1
        for core_range in core_group_2.ranges():
            for y in range(core_range.start.y, core_range.end.y + 1):
                for x in range(core_range.start.x, core_range.end.x + 1):
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        num_slices_per_core_group_2,
                        slice_offset_group2,
                    ]
                    slice_offset_group2 += num_slices_per_core_group_2

    # Set empty runtime args for idle cores
    for y in range(grid_height):
        for x in range(grid_width):
            try:
                _ = reader_rt_args[x][y]
            except (KeyError, IndexError):
                reader_rt_args[x][y] = []

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_softmax.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # CT args: Ht, Wt, HtWt, is_dim_w, TensorAccessorArgs(output)
    writer_ct_args = [Ht, Wt, HtWt, is_dim_w]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    slice_offset_group1 = 0
    for core_range in core_group_1.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    num_slices_per_core_group_1,
                    slice_offset_group1,
                ]
                slice_offset_group1 += num_slices_per_core_group_1

    if core_group_2.num_cores() > 0:
        slice_offset_group2 = slice_offset_group1
        for core_range in core_group_2.ranges():
            for y in range(core_range.start.y, core_range.end.y + 1):
                for x in range(core_range.start.x, core_range.end.x + 1):
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        num_slices_per_core_group_2,
                        slice_offset_group2,
                    ]
                    slice_offset_group2 += num_slices_per_core_group_2

    # Set empty runtime args for idle cores
    for y in range(grid_height):
        for x in range(grid_width):
            try:
                _ = writer_rt_args[x][y]
            except (KeyError, IndexError):
                writer_rt_args[x][y] = []

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_softmax.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    # CT args: Ht, Wt, NC (always 1 since batches folded into work split), is_dim_w, numeric_stable
    compute_ct_args = [Ht, Wt, 1, is_dim_w, 1 if numeric_stable else 0]

    compute_rt_args = ttnn.RuntimeArgs()
    # Compute kernel needs to know how many slices to process per core
    slice_offset_group1 = 0
    for core_range in core_group_1.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                compute_rt_args[x][y] = [num_slices_per_core_group_1]
                slice_offset_group1 += num_slices_per_core_group_1

    if core_group_2.num_cores() > 0:
        for core_range in core_group_2.ranges():
            for y in range(core_range.start.y, core_range.end.y + 1):
                for x in range(core_range.start.x, core_range.end.x + 1):
                    compute_rt_args[x][y] = [num_slices_per_core_group_2]

    # Set empty runtime args for idle cores
    for y in range(grid_height):
        for x in range(grid_width):
            try:
                _ = compute_rt_args[x][y]
            except (KeyError, IndexError):
                compute_rt_args[x][y] = []

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        cbs=[
            cb_input_descriptor,
            cb_scaler_descriptor,
            cb_out_descriptor,
            cb_max_descriptor,
            cb_exps_descriptor,
            cb_recip_sum_descriptor,
        ],
        semaphores=[],
    )
