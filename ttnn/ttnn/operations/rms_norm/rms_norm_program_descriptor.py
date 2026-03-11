# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for RMS normalization.

Work unit: tile-row (32 rows x full W, producing Wt tiles)
Total units: N * C * Ht where Ht = H / 32
"""

import struct
from pathlib import Path
from typing import Optional

import ttnn

# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for RMS normalization.

    Args:
        input_tensor: Input tensor (on device)
        output_tensor: Pre-allocated output tensor (on device)
        gamma: Optional gamma tensor (on device, ROW_MAJOR_LAYOUT)
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)
    W = shape[-1]
    H = shape[-2]

    # Compute Wt and Ht (tiles along W and H)
    Wt = W // 32
    Ht = H // 32

    # Output Wt may differ from input Wt (e.g., reduced-shape intermediate stages)
    output_W = output_tensor.shape[-1]
    output_Wt = output_W // 32

    # Total tile-rows: N * C * ... * Ht
    # For rank >= 2, batch dims are everything before last 2
    num_batch = 1
    for i in range(rank - 2):
        num_batch *= shape[i]
    total_tile_rows = num_batch * Ht

    is_input_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    fp32_dest_acc_en = input_tensor.dtype == ttnn.float32

    # Page sizes
    # All CBs use tile-sized pages for tilize/untilize compatibility
    tile_size = ttnn.tile_size(input_tensor.dtype)
    # For TensorAccessor: use actual buffer page size (stick for RM, tile for TILE)
    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    # Intermediate data format: Float32 if fp32_dest_acc_en, else same as input
    intermed_dtype = ttnn.float32 if fp32_dest_acc_en else input_tensor.dtype
    intermed_tile_size = ttnn.tile_size(intermed_dtype)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    (num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2) = ttnn.split_work_to_cores(
        compute_grid, total_tile_rows
    )

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices per design doc
    CB_IN = 0  # Input staging (RM sticks or tiles)
    CB_SCALER = 1  # Reduce scaler tile (1/W)
    CB_EPS = 2  # Epsilon scalar tile
    CB_GAMMA_RM = 3  # Gamma RM sticks (tilize staging)
    CB_GAMMA = 4  # Gamma tilized tiles
    CB_OUT = 16  # Final output
    CB_X = 24  # Tilized input (RM path)
    CB_XSQ = 25  # x^2 intermediate
    CB_RMS = 26  # Reduce output: mean(x^2)
    CB_RSQRT = 27  # rsqrt(mean + eps)
    CB_NORMED = 28  # x * rsqrt (pre-gamma)

    cbs = []

    # cb_in (c_0): Wt pages, tile-sized pages (for tilize compatibility in RM path)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_IN,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_scaler (c_1): 1 page, bfloat16 (reduce scaler requirement)
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * scaler_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # cb_eps (c_2): 1 page, bfloat16
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * scaler_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EPS,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # cb_gamma_rm (c_3): Wt pages, input data format (if gamma)
    if has_gamma:
        gamma_page_size = gamma.buffer_page_size() if not is_input_rm else tile_size
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_RM,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

        # cb_gamma (c_4): Wt pages, input data format
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # cb_out (c_16): output_Wt pages, tile-sized pages (for untilize compatibility in RM path)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=output_Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT,
                    data_format=output_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_x (c_24): Wt pages, input data format (RM path: tilized input; also reused as pre-untilize staging)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_X,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_xsq (c_25): Wt pages, intermediate format
    # Needs Wt pages because square writes all Wt tiles sequentially
    # before reduce starts consuming them (both run on same compute thread)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * intermed_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_XSQ,
                    data_format=intermed_dtype,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_rms (c_26): 1 page, intermediate format
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * intermed_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RMS,
                    data_format=intermed_dtype,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_rsqrt (c_27): 1 page, intermediate format
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * intermed_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RSQRT,
                    data_format=intermed_dtype,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # cb_normed (c_28): intermediate format
    # For no-gamma RM: Wt pages (mul_col output -> untilize input)
    # For gamma: 1 page streaming (mul_col -> mul_row)
    # For no-gamma TILE: 1 page (not used but allocated for simplicity)
    normed_pages = Wt if (is_input_rm and not has_gamma) else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=normed_pages * intermed_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_NORMED,
                    data_format=intermed_dtype,
                    page_size=intermed_tile_size,
                )
            ],
        )
    )

    # ========== 4. COMPILE-TIME AND RUNTIME ARGS ==========

    # Compile-time defines
    # NOTE: Avoid short define names like "Wt" that collide with local variable
    # names in kernel_lib headers. Use prefixed names instead.
    defines = [
        ("IS_INPUT_RM", "1" if is_input_rm else "0"),
        ("HAS_GAMMA", "1" if has_gamma else "0"),
        ("RMS_Wt", str(Wt)),
        ("ENABLE_FP32_DEST_ACC", "1" if fp32_dest_acc_en else "0"),
    ]

    # --- Reader kernel ---
    # stick_size is the page size for TensorAccessor and noc_async_read/write:
    # RM: W * elem_size (one stick = one row of W elements)
    # TILE: tile_size (one page = one tile)
    stick_size = W * input_tensor.element_size() if is_input_rm else tile_size
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    grid_width = compute_grid.x
    grid_height = compute_grid.y

    # Bit-cast epsilon float to uint32_t for passing as runtime arg
    eps_u32 = struct.unpack("I", struct.pack("f", epsilon))[0]

    current_row = 0
    for core_range in core_group_1.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                gamma_addr = gamma.buffer_address() if has_gamma else 0
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),  # src_addr
                    tiles_per_core_g1,  # num_rows
                    current_row,  # start_row_id
                    Wt,  # Wt
                    gamma_addr,  # gamma_addr
                    eps_u32,  # epsilon (float bit-cast to uint32)
                ]
                current_row += tiles_per_core_g1

    for core_range in core_group_2.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                gamma_addr = gamma.buffer_address() if has_gamma else 0
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),
                    tiles_per_core_g2,
                    current_row,
                    Wt,
                    gamma_addr,
                    eps_u32,
                ]
                current_row += tiles_per_core_g2

    # Set empty runtime args for idle cores
    _set_idle_core_args(reader_rt_args, all_cores, grid_width, grid_height)

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        defines=defines,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = []  # All compile-time info passed via defines
    compute_rt_args = ttnn.RuntimeArgs()

    current_row = 0
    for core_range in core_group_1.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                compute_rt_args[x][y] = [
                    tiles_per_core_g1,  # num_rows
                    Wt,  # Wt
                    W,  # origin_w (for scaler = 1/W)
                ]
                current_row += tiles_per_core_g1

    for core_range in core_group_2.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                compute_rt_args[x][y] = [
                    tiles_per_core_g2,
                    Wt,
                    W,
                ]
                current_row += tiles_per_core_g2

    _set_idle_core_args(compute_rt_args, all_cores, grid_width, grid_height)

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        defines=defines,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    # Writer stick_size uses output page size (may differ from input for reduced-shape stages)
    output_stick_size = output_W * output_tensor.element_size() if is_input_rm else ttnn.tile_size(output_tensor.dtype)
    writer_ct_args = [output_stick_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()

    current_row = 0
    for core_range in core_group_1.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),  # dst_addr
                    tiles_per_core_g1,  # num_rows
                    current_row,  # start_row_id
                    output_Wt,  # Wt (output tiles per row)
                ]
                current_row += tiles_per_core_g1

    for core_range in core_group_2.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    tiles_per_core_g2,
                    current_row,
                    output_Wt,
                ]
                current_row += tiles_per_core_g2

    _set_idle_core_args(writer_rt_args, all_cores, grid_width, grid_height)

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        defines=defines,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )


def _set_idle_core_args(rt_args: ttnn.RuntimeArgs, active_cores, grid_width: int, grid_height: int):
    """Set empty runtime args for all cores in grid that are not in active_cores."""
    active_set = set()
    for core_range in active_cores.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                active_set.add((x, y))

    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) not in active_set:
                rt_args[x][y] = []
