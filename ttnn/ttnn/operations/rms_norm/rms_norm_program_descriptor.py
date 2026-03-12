# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.

Work unit: tile-row (one row of Wt tiles = 32 sticks of width W)
Grid: 1D linear from compute_with_storage_grid_size
Total units: Ht_total = product(shape[:-1]) / 32

CB layout:
  c_0  - cb_input_rm:  RM sticks for tilize (RM path only)         [Wt pages]
  c_1  - cb_tilized:   Tilized input (persistent per row)           [Wt pages]
  c_2  - cb_scaler:    Reduce scaler (1/W), always bfloat16         [1 page]
  c_3  - cb_sq:        Squared tiles                                [2 pages]
  c_4  - cb_rms:       Reduce output: mean(x^2)                    [2 pages]
  c_5  - cb_eps:       Epsilon constant                             [1 page]
  c_6  - cb_rms_inv:   rsqrt(mean+eps)                             [1 page]
  c_7  - cb_gamma:     Gamma weights (tilized)                     [Wt pages]
  c_8  - cb_norm_temp: Normalize intermediate (gamma path only)    [Wt pages]
  c_16 - cb_out:       Output tiles                                 [Wt pages]
  c_17 - cb_untilized: Untilized output (RM path only)             [Wt pages]
"""

import struct
from pathlib import Path
from math import prod

import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_uint32(value: float) -> int:
    """Convert a Python float to its uint32 bit representation."""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def _float_to_bfloat16_packed(value: float) -> int:
    """Convert float to packed bfloat16 (two copies in uint32): (bf16 << 16 | bf16)."""
    float_bytes = struct.pack("f", value)
    bf16_bytes = float_bytes[2:4]  # upper 16 bits (bfloat16 = truncated float32)
    packed = int.from_bytes(bf16_bytes + bf16_bytes, byteorder="little")
    return packed


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for rms_norm.

    Args:
        input_tensor: Input tensor (on device)
        output_tensor: Pre-allocated output tensor (on device)
        gamma: Optional gamma tensor (on device, ROW_MAJOR_LAYOUT)
        epsilon: Epsilon for numerical stability

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)
    W = shape[-1]
    input_dtype = input_tensor.dtype
    input_layout = input_tensor.layout
    is_rm = input_layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None

    # Tile dimensions
    TILE_H = 32
    TILE_W = 32
    Wt = W // TILE_W  # Number of tiles in width dimension

    # Total tile-rows: product of all dims except last, divided by tile height
    dims_except_last = [shape[i] for i in range(rank - 1)]
    Ht_total = prod(dims_except_last) // TILE_H

    # Tile size in bytes (for TILE_LAYOUT pages)
    tile_size = ttnn.tile_size(input_dtype)

    # Element size in bytes
    element_size = input_tensor.element_size()

    # Stick size for RM path (W elements * element_size)
    stick_size = W * element_size

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    # Use split_work_to_cores for balanced distribution
    # Work unit = tile-row (Ht_total total)
    max_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    all_cores_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (
        num_cores,
        core_grid,
        core_group_1,
        core_group_2,
        rows_per_core_group_1,
        rows_per_core_group_2,
    ) = ttnn.split_work_to_cores(all_cores_range, Ht_total)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices per design doc
    CB_INPUT_RM = 0  # RM sticks for tilize (RM path) / unused (TILE path)
    CB_TILIZED = 1  # Tilized input, persistent per row
    CB_SCALER = 2  # Reduce scaler (1/W), always bfloat16
    CB_SQ = 3  # Squared tiles (double-buffered)
    CB_RMS = 4  # Reduce output: mean(x^2) (double-buffered)
    CB_EPS = 5  # Epsilon constant
    CB_RMS_INV = 6  # rsqrt(mean+eps)
    CB_GAMMA = 7  # Gamma weights (tilized)
    CB_OUT = 16  # Output tiles
    CB_UNTILIZED = 17  # Untilized output (RM path)

    cbs = []

    # c_0: cb_input_rm - RM sticks in tile-sized pages (for tilize)
    # Needs Wt pages for: RM input tilize OR gamma loading (gamma is always RM)
    # Even for TILE path with gamma, c_0 is used for gamma stick loading before tilize
    cb0_pages = Wt if (is_rm or has_gamma) else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb0_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_RM,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: cb_tilized - Tilized input tiles (Wt pages, persistent per row)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_TILIZED,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_2: cb_scaler - Reduce scaler (1/W), matches input dtype
    # Must match input dtype so compute_kernel_hw_startup sees uniform formats
    # for srcA and srcB; a bfloat16 srcB with fp32 srcA leaves HW state that
    # corrupts pack_untilize faces 2/3 after fast_tilize for fp32+Wt>1.
    scaler_tile_size = tile_size
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=input_dtype,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # c_3: cb_sq - Squared tiles (Wt pages, square fills before reduce consumes)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SQ,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_4: cb_rms - Reduce output (2 pages, no longer reused for gamma mul output)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RMS,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_5: cb_eps - Epsilon constant (1 page)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EPS,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_6: cb_rms_inv - rsqrt(mean+eps) (1 page)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RMS_INV,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_7: cb_gamma - Gamma weights tilized (Wt pages if gamma, else 1 minimal)
    cb7_pages = Wt if has_gamma else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb7_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_GAMMA,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_8: cb_norm_temp - Normalize intermediate when gamma active (Wt pages)
    CB_NORM_TEMP = 8
    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_NORM_TEMP,
                        data_format=input_dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_16: cb_out - Output tiles (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_17: cb_untilized - Untilized output sticks (RM path only)
    cb17_pages = Wt if is_rm else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb17_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_UNTILIZED,
                    data_format=input_dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL COMPILE-TIME AND RUNTIME ARGS ==========

    # -- Scaler: 1.0/W as bfloat16 packed (bf16 << 16 | bf16) --
    scaler_value = 1.0 / float(W)
    scaler_bits = _float_to_bfloat16_packed(scaler_value)

    # -- Epsilon as float32 bits --
    eps_bits = _float_to_uint32(epsilon)

    # -- Reader kernel --
    reader_ct_args = [
        stick_size if is_rm else tile_size,  # 0: stick_size or tile_size
        scaler_bits,  # 1: scaler_bits (1/W packed bf16)
        eps_bits,  # 2: eps_bits (float32 bits)
        1 if is_rm else 0,  # 3: input_is_rm
        1 if has_gamma else 0,  # 4: has_gamma
        Wt,  # 5: Wt (tiles in width)
        stick_size,  # 6: gamma_stick_size (W * element_size, for gamma RM accessor page size)
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Gamma TensorAccessor args — always appended ([0] placeholder when absent)
    # This ensures TensorAccessorArgs offsets are stable regardless of has_gamma
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args() if has_gamma else [0])

    # -- Writer kernel --
    output_W = output_tensor.shape[-1]
    output_stick_size = output_W * element_size
    output_Wt = output_W // TILE_W
    writer_ct_args = [
        output_stick_size if is_rm else tile_size,  # 0: stick_size or tile_size
        1 if is_rm else 0,  # 1: input_is_rm
        output_Wt,  # 2: Wt (output tiles in width)
        1 if has_gamma else 0,  # 3: has_gamma
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # -- Compute kernel --
    # Compute CT args use max of the two core groups for Ht (each core gets its own count via RT args)
    # But actually, compute CT args set the max Ht per core; we use the larger group
    compute_ct_args = [
        max(rows_per_core_group_1, rows_per_core_group_2 if rows_per_core_group_2 > 0 else 0),  # 0: Ht (max per core)
        Wt,  # 1: Wt
        1 if is_rm else 0,  # 2: input_is_rm
        1 if has_gamma else 0,  # 3: has_gamma
    ]

    # ========== 5. RUNTIME ARGS (per-core) ==========
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    # Iterate over all cores in the compute grid and assign work
    current_row = 0

    # Helper to iterate core ranges
    def iterate_core_group(core_group, rows_per_core):
        nonlocal current_row
        for cr in core_group.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    if is_rm:
                        # Reader: start stick in input tensor
                        reader_start = current_row * TILE_H
                        # Writer: start stick in output tensor (same H, different W)
                        writer_start = current_row * TILE_H
                    else:
                        # Reader: start tile in input tensor
                        reader_start = current_row * Wt
                        # Writer: start tile in output tensor (uses output_Wt)
                        writer_start = current_row * output_Wt

                    reader_rt_args[x][y] = [
                        input_addr,
                        reader_start,
                        rows_per_core,
                        gamma_addr,
                    ]
                    writer_rt_args[x][y] = [
                        output_addr,
                        writer_start,
                        rows_per_core,
                    ]
                    # Compute runtime args: actual Ht for this core
                    compute_rt_args[x][y] = [rows_per_core]

                    current_row += rows_per_core

    iterate_core_group(core_group_1, rows_per_core_group_1)
    if rows_per_core_group_2 > 0:
        iterate_core_group(core_group_2, rows_per_core_group_2)

    # Set empty runtime args for idle cores (cores in the grid but not assigned work)
    for y in range(compute_grid.y):
        for x in range(compute_grid.x):
            # Only set if not already assigned
            try:
                _ = reader_rt_args[x][y]
            except (KeyError, IndexError):
                reader_rt_args[x][y] = []
                writer_rt_args[x][y] = []
                compute_rt_args[x][y] = []

    # ========== 6. KERNEL DESCRIPTORS ==========
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute config: enable FP32 accumulation if input is float32
    fp32_acc = input_dtype == ttnn.float32
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=fp32_acc,
            math_approx_mode=False,
        ),
    )

    # ========== 7. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
