# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Softmax Program Descriptor.

Defines circular buffers, kernel args, and work distribution for the
4-phase numerically-stable softmax pipeline.

Layout support:
  - TILE_LAYOUT: reader reads tiles directly into cb_input_tiles.
  - ROW_MAJOR_LAYOUT: reader reads RM sticks into cb_rm_in, compute
    tilizes into cb_input_tiles, runs the 4-phase softmax math, then
    untilizes cb_output_tiles into cb_rm_out, writer writes RM sticks.

The multi-core work distribution (split_work_to_cores over NC slabs)
is identical for both layouts — each core processes an independent set
of (N,C) slabs.

Two execution variations selected by L1 budget:
  - V1 (fast path): full-slab CBs. Used when per-core CB footprint ≤ 256 KiB.
  - V2 (streaming): constant-bounded CBs (BLOCK_SIZE). Used for wide/tall shapes.
"""

from pathlib import Path
import math
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# L1 budget threshold for V1 vs V2 dispatch (256 KiB per the prompt)
V1_CB_BUDGET = 256 * 1024

# Block size for V2 streaming path (constant, not dependent on Wt/Ht)
V2_BLOCK_SIZE = 4  # tiles per chunk along the reduce dimension


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    dim: int = -1,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.ProgramDescriptor:
    """Create the ProgramDescriptor for softmax.

    Args:
        input_tensor: Input tensor (on device, TILE or ROW_MAJOR layout)
        output_tensor: Pre-allocated output tensor (on device)
        dim: Canonicalized dimension (-1 or -2)
        compute_kernel_config: Compute kernel config (math_fidelity, fp32_dest_acc_en, ...)
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    input_shape = list(input_tensor.shape)
    N, C = input_shape[0], input_shape[1]
    H, W = input_shape[2], input_shape[3]
    Ht = (H + TILE_DIM - 1) // TILE_DIM
    Wt = (W + TILE_DIM - 1) // TILE_DIM
    NC = N * C  # number of slabs

    origin_W = W
    origin_H = H
    partial_W = W % TILE_DIM
    partial_H = H % TILE_DIM
    has_partial = (partial_W > 0) if dim == -1 else (partial_H > 0)
    num_scaler_tiles = 2 if has_partial else 1

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)
    intermediate_tile_size = ttnn.tile_size(ttnn.float32)
    input_tile_size = ttnn.tile_size(input_tensor.dtype)
    output_tile_size = ttnn.tile_size(output_tensor.dtype)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    device_info = ttnn._ttnn.reports.get_device_info(device)
    num_cores_x = device_info.num_x_compute_cores
    num_cores_y = device_info.num_y_compute_cores

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
    ) = ttnn.split_work_to_cores(ttnn.CoreCoord(num_cores_x, num_cores_y), NC)

    # ========== 2.5 V1 vs V2 DISPATCH ==========
    reduce_dim_tiles = Wt if dim == -1 else Ht  # Wt for REDUCE_ROW, Ht for REDUCE_COL
    non_reduce_dim = Ht if dim == -1 else Wt
    tiles_per_slab = Ht * Wt

    # Compute V1 per-core CB footprint (sum of all CB sizes)
    v1_cb_footprint = (
        tiles_per_slab * input_tile_size  # cb_input_tiles
        + num_scaler_tiles * scaler_tile_size  # cb_scaler_max
        + num_scaler_tiles * scaler_tile_size  # cb_scaler_sum
        + (tiles_per_slab * output_tile_size if not is_rm else 0)  # cb_output_tiles (TILE: full slab)
        + (2 * Wt * input_tile_size if is_rm else 0)  # cb_rm_in (RM only, double-buffered)
        + (2 * Wt * output_tile_size if is_rm else 0)  # cb_rm_out (RM only, double-buffered)
        + reduce_dim_tiles * intermediate_tile_size  # cb_max
        + tiles_per_slab * intermediate_tile_size  # cb_exp
        + reduce_dim_tiles * intermediate_tile_size  # cb_recip_sum
    )
    # For RM path, cb_output_tiles is full slab (untilize can't pipeline)
    if is_rm:
        v1_cb_footprint += tiles_per_slab * output_tile_size

    use_v2 = v1_cb_footprint > V1_CB_BUDGET

    # V2 has two sub-modes:
    #  - chunk_along_reduce: 3-pass approach, chunks the reduce dim.
    #    Requires reduce_dim_tiles >= BLOCK_SIZE and divisible.
    #  - chunk_along_non_reduce: V1-style 4-phase per chunk, chunks the
    #    non-reduce dim. Used when reduce dim is too small to chunk.
    chunk_along_reduce = use_v2 and reduce_dim_tiles >= V2_BLOCK_SIZE and reduce_dim_tiles % V2_BLOCK_SIZE == 0
    chunk_along_non_reduce = use_v2 and not chunk_along_reduce

    block_size = V2_BLOCK_SIZE if use_v2 else 0

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    CB_INPUT_TILES = 0
    CB_SCALER_MAX = 1
    CB_SCALER_SUM = 2
    CB_RM_IN = 3
    CB_OUTPUT_TILES = 16
    CB_RM_OUT = 17
    CB_MAX = 24
    CB_EXP = 25
    CB_RECIP_SUM = 26
    # V2-only CBs
    CB_RUNNING_MAX = 27
    CB_RUNNING_SUM = 28
    CB_RECIP_SUM_V2 = 29
    CB_CHUNK_MAX = 30
    CB_CHUNK_SUM = 31

    cbs = []

    if not use_v2:
        # ===== V1 CBs (full-slab, same as before) =====
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tiles_per_slab * input_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=input_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_scaler_tiles * scaler_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_SCALER_MAX, data_format=ttnn.bfloat16, page_size=scaler_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_scaler_tiles * scaler_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_SCALER_SUM, data_format=ttnn.bfloat16, page_size=scaler_tile_size
                    )
                ],
            )
        )
        if is_rm:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * Wt * input_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_RM_IN, data_format=input_tensor.dtype, page_size=input_tile_size
                        )
                    ],
                )
            )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tiles_per_slab * output_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_OUTPUT_TILES, data_format=output_tensor.dtype, page_size=output_tile_size
                    )
                ],
            )
        )
        if is_rm:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * Wt * output_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_RM_OUT, data_format=output_tensor.dtype, page_size=output_tile_size
                        )
                    ],
                )
            )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=reduce_dim_tiles * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_MAX, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tiles_per_slab * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_EXP, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=reduce_dim_tiles * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RECIP_SUM, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
    else:
        # ===== V2 CBs (constant-bounded) =====
        # chunk_along_reduce: CBs are BLOCK_SIZE tiles (3-pass, chunks reduce dim)
        # chunk_along_non_reduce: CBs are BLOCK_SIZE * reduce_dim_tiles tiles
        #   (V1-style 4-phase per chunk, chunks non-reduce dim)
        chunk_tiles = block_size if chunk_along_reduce else block_size * reduce_dim_tiles
        chunk_output_tiles = chunk_tiles  # same tile count for output

        # cb_input_tiles: chunk_tiles (reader→compute or tilize→compute)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=chunk_tiles * input_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=input_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_scaler_tiles * scaler_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_SCALER_MAX, data_format=ttnn.bfloat16, page_size=scaler_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_scaler_tiles * scaler_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_SCALER_SUM, data_format=ttnn.bfloat16, page_size=scaler_tile_size
                    )
                ],
            )
        )
        if is_rm:
            # cb_rm_in: double-buffered, chunk_tiles tile-pages
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * chunk_tiles * input_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_RM_IN, data_format=input_tensor.dtype, page_size=input_tile_size
                        )
                    ],
                )
            )
        # cb_output_tiles: chunk_tiles (streaming compute→writer)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=chunk_output_tiles * output_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_OUTPUT_TILES, data_format=output_tensor.dtype, page_size=output_tile_size
                    )
                ],
            )
        )
        if is_rm:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * chunk_output_tiles * output_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_RM_OUT, data_format=output_tensor.dtype, page_size=output_tile_size
                        )
                    ],
                )
            )
        # cb_exp: chunk_tiles (intermediate)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=chunk_tiles * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_EXP, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        # V2 persistent stats (1 tile each, fp32)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=1 * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RUNNING_MAX, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=1 * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RUNNING_SUM, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=1 * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RECIP_SUM_V2, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=1 * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_CHUNK_MAX, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=1 * intermediate_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_CHUNK_SUM, data_format=ttnn.float32, page_size=intermediate_tile_size
                    )
                ],
            )
        )
        # For chunk_along_non_reduce: need cb_max (24) and cb_recip_sum (26)
        # sized to BLOCK_SIZE tiles (one per non-reduce element in the chunk)
        if chunk_along_non_reduce:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=block_size * intermediate_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_MAX, data_format=ttnn.float32, page_size=intermediate_tile_size
                        )
                    ],
                )
            )
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=block_size * intermediate_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=CB_RECIP_SUM, data_format=ttnn.float32, page_size=intermediate_tile_size
                        )
                    ],
                )
            )

    # ========== 4. KERNEL DESCRIPTORS ==========
    cores = ttnn.grid_to_cores(num_cores, num_cores_x, num_cores_y, row_wise=False)
    num_cores_group_1 = core_group_1.num_cores()

    is_rm_flag = 1 if is_rm else 0

    units_per_slab = H if is_rm else tiles_per_slab

    # Select kernel files based on V1/V2
    if not use_v2:
        reader_source = str(KERNEL_DIR / "softmax_reader.cpp")
        writer_source = str(KERNEL_DIR / "softmax_writer.cpp")
        compute_source = str(KERNEL_DIR / "softmax_compute.cpp")
        # CT args: Ht, Wt, dim, is_rm, origin_W, origin_H
        reader_ct_args = [Ht, Wt, dim & 0xFFFFFFFF, is_rm_flag, origin_W, origin_H]
        writer_ct_args = [Ht, Wt, is_rm_flag, origin_W, origin_H]
        compute_ct_args = [Ht, Wt, dim & 0xFFFFFFFF, is_rm_flag, origin_W, origin_H]
    else:
        reader_source = str(KERNEL_DIR / "softmax_reader_v2.cpp")
        writer_source = str(KERNEL_DIR / "softmax_writer_v2.cpp")
        compute_source = str(KERNEL_DIR / "softmax_compute_v2.cpp")
        # CT args: Ht, Wt, dim, is_rm, origin_W, origin_H, BLOCK_SIZE, chunk_along_reduce
        chunk_along_reduce_flag = 1 if chunk_along_reduce else 0
        reader_ct_args = [Ht, Wt, dim & 0xFFFFFFFF, is_rm_flag, origin_W, origin_H, block_size, chunk_along_reduce_flag]
        writer_ct_args = [
            Ht,
            Wt,
            is_rm_flag,
            origin_W,
            origin_H,
            dim & 0xFFFFFFFF,
            block_size,
            chunk_along_reduce_flag,
        ]
        compute_ct_args = [
            Ht,
            Wt,
            dim & 0xFFFFFFFF,
            is_rm_flag,
            origin_W,
            origin_H,
            block_size,
            chunk_along_reduce_flag,
        ]

    # --- Reader kernel ---
    reader_ct_args_full = list(reader_ct_args)
    reader_ct_args_full.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    start_offset = 0
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        reader_rt_args[core.x][core.y] = [input_tensor.buffer_address(), start_offset, slabs_per_core]
        start_offset += slabs_per_core * units_per_slab

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=reader_source,
        core_ranges=all_cores,
        compile_time_args=reader_ct_args_full,
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    reader_kernel.runtime_args = reader_rt_args

    # --- Writer kernel ---
    writer_ct_args_full = list(writer_ct_args)
    writer_ct_args_full.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    start_offset = 0
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), start_offset, slabs_per_core]
        start_offset += slabs_per_core * units_per_slab

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=writer_source,
        core_ranges=all_cores,
        compile_time_args=writer_ct_args_full,
        runtime_args=[],
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_kernel.runtime_args = writer_rt_args

    # --- Compute kernel ---
    compute_rt_args = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        compute_rt_args[core.x][core.y] = [slabs_per_core]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_source,
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=compute_kernel_config,
    )
    compute_kernel.runtime_args = compute_rt_args

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
