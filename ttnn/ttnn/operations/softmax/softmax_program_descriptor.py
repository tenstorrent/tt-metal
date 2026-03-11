# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax Operation - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for the softmax operation.

CB layout:
    c_0  (input):     2 tiles, bf16 - input tiles from DRAM
    c_1  (scaler):    1 tile,  bf16 - reduce scaler (all 1.0)
    c_2  (mm_scaler): 1 tile,  bf16 - matmul row-reduce scaler
    c_3  (max):       2 tiles, bf16 - max reduction output
    c_4  (exp_sum):   2 tiles, bf16 - sum of exp output
    c_5  (recip_sum): 2 tiles, bf16 - 1/sum
    c_16 (output):    2 tiles, bf16 - output tiles to DRAM
"""

from pathlib import Path
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices
CB_INPUT = 0
CB_SCALER = 1
CB_MM_SCALER = 2
CB_MAX = 3
CB_EXP_SUM = 4
CB_RECIP_SUM = 5
CB_OUTPUT = 16


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the softmax operation.

    Args:
        input_tensor: Input tensor (on device, bfloat16, TILE_LAYOUT)
        output_tensor: Pre-allocated output tensor (on device)
        dim: Reduction dimension (-1 for width, -2 for height)
        numeric_stable: Whether to subtract max before exp

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    # Tile dimensions
    TILE_H = 32
    TILE_W = 32
    Ht = H // TILE_H
    Wt = W // TILE_W

    # Total tiles
    num_tiles = N * C * Ht * Wt

    # Page size from tensor metadata
    page_size = input_tensor.buffer_page_size()

    # ========== 2. CORE GRID (single core) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # c_0: input (2 tiles, double-buffered)
    cb_input_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # c_1: scaler (1 tile, persistent constant)
    cb_scaler_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # c_2: mm_scaler (1 tile, persistent constant)
    cb_mm_scaler_desc = ttnn.CBDescriptor(
        total_size=1 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MM_SCALER,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # c_3: max (2 tiles, double-buffered for compute self-production/consumption)
    cb_max_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MAX,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # c_4: exp_sum (2 tiles)
    cb_exp_sum_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EXP_SUM,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # c_5: recip_sum (2 tiles)
    cb_recip_sum_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RECIP_SUM,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # c_16: output (2 tiles, double-buffered)
    cb_output_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT,
                data_format=output_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # ========== 4. WORK DISTRIBUTION ==========
    # dim=-1: reduce along width, iterate over rows (NC * Ht rows, each Wt tiles wide)
    # dim=-2: reduce along height, iterate over columns (NC * Wt columns, each Ht tiles tall)
    if dim == -1:
        num_rows_or_cols = N * C * Ht  # number of rows to process
        inner_dim = Wt  # tiles along reduce dimension
    else:  # dim == -2
        num_rows_or_cols = N * C * Wt  # number of columns to process
        inner_dim = Ht  # tiles along reduce dimension

    # ========== 5. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [Wt, Ht]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_softmax.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [num_rows_or_cols, inner_dim]

    # Preprocessor defines for compute kernel
    compute_defines = [
        ("REDUCE_OP", "PoolType::MAX"),
        ("REDUCE_DIM", "ReduceDim::REDUCE_ROW" if dim == -1 else "ReduceDim::REDUCE_COL"),
    ]
    if numeric_stable:
        compute_defines.append(("NUMERIC_STABLE", "1"))
    else:
        compute_defines.append(("NUMERIC_STABLE", "0"))

    if dim == -1:
        compute_defines.append(("DIM_W", "1"))
    else:
        compute_defines.append(("DIM_H", "1"))

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_softmax.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        defines=compute_defines,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [num_tiles]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_softmax.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 6. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[
            cb_input_desc,
            cb_scaler_desc,
            cb_mm_scaler_desc,
            cb_max_desc,
            cb_exp_sum_desc,
            cb_recip_sum_desc,
            cb_output_desc,
        ],
    )
