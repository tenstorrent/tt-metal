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
"""

from pathlib import Path
import math
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32


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
    Ht = H // TILE_DIM
    Wt = W // TILE_DIM
    NC = N * C  # number of slabs

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    # Scaler tiles are always bfloat16 (reduce scaler convention)
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Intermediate accumulator CBs must be Float32 — fp32_dest_acc_en is
    # always True (the op is fp32-dest-only), so accumulations cross the
    # CB at full fp32 precision.
    intermediate_tile_size = ttnn.tile_size(ttnn.float32)

    # Tile-size for the input/output dtype (used for tiled CBs)
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

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices: 0-7 input, 8-15 special, 16-23 output, 24-31 intermediate
    CB_INPUT_TILES = 0  # tiled input (TILE: reader→compute; RM: tilize→compute)
    CB_SCALER_MAX = 1
    CB_SCALER_SUM = 2
    CB_RM_IN = 3  # RM sticks input (RM path only: reader→tilize)
    CB_OUTPUT_TILES = 16  # tiled output (compute→writer for TILE; compute→untilize for RM)
    CB_RM_OUT = 17  # RM sticks output (RM path only: untilize→writer)
    CB_MAX = 24
    CB_EXP = 25
    CB_RECIP_SUM = 26

    reduce_dim_tiles = Ht if dim == -1 else Wt  # Ht for REDUCE_ROW, Wt for REDUCE_COL
    tiles_per_slab = Ht * Wt

    cbs = []

    # --- cb_input_tiles: tiled input data ---
    # TILE path: reader pushes full slab (Ht*Wt tiles).
    # RM path: compute (tilize) pushes full slab (Ht*Wt tiles).
    # Both paths use the same CB; only the producer differs (reader vs compute).
    # Since only one path is active per kernel compilation (CT-arg dispatch),
    # the single-producer rule (§2.2) is respected.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tiles_per_slab * input_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=input_tile_size,
                )
            ],
        )
    )

    # --- cb_scaler_max: 1 page, bf16 ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * scaler_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER_MAX,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # --- cb_scaler_sum: 1 page, bf16 ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * scaler_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER_SUM,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # --- cb_rm_in: RM sticks input (RM path only) ---
    # Reader pushes sticks via read_sticks_for_tilize (TILE granularity).
    # With TILE granularity, the CB page_size = tile_size and each call
    # produces Wt tile-sized pages from 32 sticks.
    # Double-buffered (2*Wt pages) so reader and tilize can pipeline.
    if is_rm:
        rm_in_double_buffer = 2
        rm_in_total_size = rm_in_double_buffer * Wt * input_tile_size
        cbs.append(
            ttnn.CBDescriptor(
                total_size=rm_in_total_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RM_IN,
                        data_format=input_tensor.dtype,
                        page_size=input_tile_size,
                    )
                ],
            )
        )

    # --- cb_output_tiles: tiled output data ---
    # TILE path: compute pushes tiles; writer pops them.
    # RM path: compute pushes full slab (Ht*Wt tiles); untilize consumes.
    # Sized for the max of both: full slab (RM needs it, TILE path is fine with more).
    output_cb_total = tiles_per_slab * output_tile_size
    cbs.append(
        ttnn.CBDescriptor(
            total_size=output_cb_total,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=output_tile_size,
                )
            ],
        )
    )

    # --- cb_rm_out: RM sticks output (RM path only) ---
    # Untilize pushes sticks via write_sticks_after_untilize.
    # Untilize always produces tile-sized pages on its output CB.
    # Double-buffered (2*Wt pages) so untilize and writer can pipeline.
    if is_rm:
        rm_out_double_buffer = 2
        rm_out_total_size = rm_out_double_buffer * Wt * output_tile_size
        cbs.append(
            ttnn.CBDescriptor(
                total_size=rm_out_total_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RM_OUT,
                        data_format=output_tensor.dtype,
                        page_size=output_tile_size,
                    )
                ],
            )
        )

    # --- cb_max: full reduce-dim block, Float32 (accumulator intermediate) ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=reduce_dim_tiles * intermediate_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MAX,
                    data_format=ttnn.float32,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # --- cb_exp: full slab, Float32 (accumulator intermediate) ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tiles_per_slab * intermediate_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EXP,
                    data_format=ttnn.float32,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # --- cb_recip_sum: full reduce-dim block, Float32 (accumulator intermediate) ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=reduce_dim_tiles * intermediate_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RECIP_SUM,
                    data_format=ttnn.float32,
                    page_size=intermediate_tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========
    cores = ttnn.grid_to_cores(num_cores, num_cores_x, num_cores_y, row_wise=False)
    num_cores_group_1 = core_group_1.num_cores()

    is_rm_flag = 1 if is_rm else 0

    # Work-unit offset per core:
    #   TILE: offset is in tile units (tiles_per_slab = Ht * Wt per slab)
    #   RM:   offset is in stick (page) units (sticks_per_slab = H per slab)
    # Both advance by slabs_per_core * units_per_slab.
    units_per_slab = H if is_rm else tiles_per_slab  # H = Ht * 32 sticks, or Ht*Wt tiles

    # --- Reader kernel ---
    # CT args: Ht, Wt, dim, is_rm (4 scalar), then TensorAccessorArgs
    reader_ct_args = [Ht, Wt, dim & 0xFFFFFFFF, is_rm_flag]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    start_offset = 0
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        reader_rt_args[core.x][core.y] = [
            input_tensor.buffer_address(),
            start_offset,  # start_tile_id (TILE) or start_stick_id (RM)
            slabs_per_core,
        ]
        start_offset += slabs_per_core * units_per_slab

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    reader_kernel.runtime_args = reader_rt_args

    # --- Writer kernel ---
    # CT args: Ht, Wt, is_rm (3 scalar), then TensorAccessorArgs
    writer_ct_args = [Ht, Wt, is_rm_flag]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    start_offset = 0
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        writer_rt_args[core.x][core.y] = [
            output_tensor.buffer_address(),
            start_offset,  # start_tile_id (TILE) or start_stick_id (RM)
            slabs_per_core,
        ]
        start_offset += slabs_per_core * units_per_slab

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=[],
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_kernel.runtime_args = writer_rt_args

    # --- Compute kernel ---
    # CT args: Ht, Wt, dim, is_rm (4 scalar)
    compute_ct_args = [Ht, Wt, dim & 0xFFFFFFFF, is_rm_flag]

    compute_rt_args = ttnn.RuntimeArgs()
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        compute_rt_args[core.x][core.y] = [slabs_per_core]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=compute_kernel_config,
    )
    compute_kernel.runtime_args = compute_rt_args

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
