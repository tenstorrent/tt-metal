# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Softmax Program Descriptor.

Defines circular buffers, kernel args, and work distribution for the
4-phase numerically-stable softmax pipeline.
"""

from pathlib import Path
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
        input_tensor: Input tensor (on device, TILE_LAYOUT, float32)
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

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    # Scaler tiles are always bfloat16 (reduce scaler convention)
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Intermediate accumulator CBs must be Float32 — fp32_dest_acc_en is
    # always True (the op is fp32-dest-only), so accumulations cross the
    # CB at full fp32 precision.  Using the input/output dtype here would
    # truncate the accumulator at each phase boundary (pack_tile rounds
    # to the CB's format), erasing the fp32-dest gain.
    intermediate_tile_size = ttnn.tile_size(ttnn.float32)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Use the device's compute grid, capped at NC slabs
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
    CB_INPUT_TILES = 0
    CB_SCALER_MAX = 1
    CB_SCALER_SUM = 2
    CB_OUTPUT_TILES = 16
    CB_MAX = 24
    CB_EXP = 25
    CB_RECIP_SUM = 26

    # Sizing per the design:
    # cb_input_tiles: Ht×Wt pages (full slab; WaitUpfrontNoPop requires entire slab)
    # cb_scaler_max/sum: 1 page each (constant, never popped)
    # cb_max: Ht (dim=-1) or Wt (dim=-2) pages (sequential helper intermediate)
    # cb_exp: Ht×Wt pages (full block; WaitUpfrontNoPop + Bulk consumer)
    # cb_recip_sum: Ht (dim=-1) or Wt (dim=-2) pages (sequential helper intermediate)
    # cb_output_tiles: 2 pages (streaming, double-buffered)

    reduce_dim_tiles = Ht if dim == -1 else Wt  # Ht for REDUCE_ROW, Wt for REDUCE_COL
    tiles_per_slab = Ht * Wt

    cbs = [
        # cb_input_tiles: full slab, single-buffered
        ttnn.CBDescriptor(
            total_size=tiles_per_slab * input_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        ),
        # cb_scaler_max: 1 page, bf16
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
        ),
        # cb_scaler_sum: 1 page, bf16
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
        ),
        # cb_output_tiles: 2 pages, fp32, double-buffered streaming
        ttnn.CBDescriptor(
            total_size=2 * output_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=output_page_size,
                )
            ],
        ),
        # cb_max: full reduce-dim block, Float32 (accumulator intermediate)
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
        ),
        # cb_exp: full slab, Float32 (accumulator intermediate)
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
        ),
        # cb_recip_sum: full reduce-dim block, Float32 (accumulator intermediate)
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
        ),
    ]

    # ========== 4. KERNEL DESCRIPTORS ==========
    # Enumerate cores for runtime args assignment
    cores = ttnn.grid_to_cores(num_cores, num_cores_x, num_cores_y, row_wise=False)

    # Number of cores in each group — CoreRangeSet doesn't support len()
    num_cores_group_1 = core_group_1.num_cores()

    # --- Reader kernel ---
    # CT args: Ht, Wt, dim (3 scalar), then TensorAccessorArgs
    # dim is cast to uint32 (two's complement: -1 → 0xFFFFFFFF, -2 → 0xFFFFFFFE)
    # C++ side reads it back via get_compile_time_arg_val<uint32_t> and casts to int32_t
    reader_ct_args = [Ht, Wt, dim & 0xFFFFFFFF]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    tile_offset = 0
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        reader_rt_args[core.x][core.y] = [
            input_tensor.buffer_address(),
            tile_offset,  # start_tile_id
            slabs_per_core,
        ]
        tile_offset += slabs_per_core * tiles_per_slab

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    reader_kernel.runtime_args = reader_rt_args

    # --- Writer kernel ---
    # CT args: Ht, Wt (2 scalar), then TensorAccessorArgs
    writer_ct_args = [Ht, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    tile_offset = 0
    for i, core in enumerate(cores):
        slabs_per_core = units_per_core_group_1 if i < num_cores_group_1 else units_per_core_group_2
        writer_rt_args[core.x][core.y] = [
            output_tensor.buffer_address(),
            tile_offset,  # start_tile_id
            slabs_per_core,
        ]
        tile_offset += slabs_per_core * tiles_per_slab

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=[],
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_kernel.runtime_args = writer_rt_args

    # --- Compute kernel ---
    # CT args: Ht, Wt, dim (needed for template dispatch at compile time)
    # num_slabs is a runtime arg (varies per core)
    compute_ct_args = [Ht, Wt, dim & 0xFFFFFFFF]

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
