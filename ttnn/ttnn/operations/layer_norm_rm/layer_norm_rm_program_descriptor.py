# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for row-major layer normalization.

Work unit: tile-row (one row of Wt tiles = 32 RM sticks).
Reader reads RM sticks, compute tilizes/normalizes/untilizes, writer writes RM sticks.
"""

import struct
from pathlib import Path

import ttnn

# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_bfloat16_packed(value: float) -> int:
    """Convert a float to packed bfloat16 format: (bf16 << 16 | bf16).

    This is the standard format for scaler tiles on Tenstorrent hardware.
    The bfloat16 value is duplicated in both the upper and lower 16-bit halves.
    """
    float_bytes = struct.pack("f", value)
    # bfloat16 is the upper 16 bits of float32
    bf16_bits = int.from_bytes(float_bytes[2:4], byteorder="little")
    return (bf16_bits << 16) | bf16_bits


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor: Input tensor (bfloat16, ROW_MAJOR_LAYOUT, on device)
        output_tensor: Pre-allocated output tensor (bfloat16, ROW_MAJOR_LAYOUT, on device)
        gamma: Optional scale tensor (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT
        beta: Optional shift tensor (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    W = shape[-1]
    H = shape[-2]
    # Total number of rows across all batch/channel dims
    num_dims = len(shape)
    total_rows = 1
    for i in range(num_dims - 1):
        total_rows *= shape[i]
    total_rows *= (
        H  # H is the second-to-last dim but already factored; correct: total_rows = product of all dims except W
    )
    # Actually: shape is (N, C, H, W) or (H, W) etc. total_rows = product of all dims except W
    total_rows = 1
    for i in range(num_dims - 1):
        total_rows *= shape[i]

    Wt = W // 32  # tiles per row
    Ht_total = total_rows // 32  # total tile-rows (each tile-row = 32 RM sticks)

    # tile_size for bfloat16 32x32 tile
    tile_size = ttnn.tile_size(ttnn.bfloat16)

    # stick_size = W * element_size (2 bytes for bf16)
    stick_size = W * 2

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    # Use split_work_to_cores to distribute tile-rows across cores
    # We need at least 2 cores for split_work_to_cores to work properly
    max_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tile_rows_per_core_g1,
        tile_rows_per_core_g2,
    ) = ttnn.split_work_to_cores(full_grid, Ht_total)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices per design:
    #   c_0:  input RM sticks (reader -> compute tilize)
    #   c_1:  tilized input (compute internal, double-buffered)
    #   c_2:  gamma RM sticks (reader -> compute tilize, once)
    #   c_3:  beta RM sticks (reader -> compute tilize, once)
    #   c_8:  reduce scaler tile (reader -> compute, persistent)
    #   c_9:  epsilon constant tile (reader -> compute, persistent)
    #   c_16: output RM sticks (compute untilize -> writer)
    #   c_24: mean tile (compute internal)
    #   c_25: centered values (compute internal, double-buffered for reuse)
    #   c_26: squared centered / reused for gamma*norm (compute internal)
    #   c_27: variance tile (compute internal)
    #   c_28: rstd tile (compute internal)
    #   c_29: normalized output (compute internal)
    #   c_30: tilized gamma (compute internal, persistent)
    #   c_31: tilized beta (compute internal, persistent)

    cb_descriptors = []

    # c_0: input RM sticks - Wt pages of tile_size
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: tilized input - 2*Wt pages (double-buffered for reuse with NoWaitNoPop)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=2 * Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=1,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_2: gamma RM sticks - Wt pages (only if gamma)
    if has_gamma:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=2,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_3: beta RM sticks - Wt pages (only if beta)
    if has_beta:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=3,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_8: reduce scaler - 1 tile (persistent)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=8,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_9: epsilon constant - 1 tile (persistent)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=9,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_16: output RM sticks - Wt pages
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=16,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_24: mean tile - 1 tile
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=24,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_25: centered values - 2*Wt tiles (double-buffered for reuse)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=2 * Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=25,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_26: squared centered / gamma*norm reuse - Wt tiles
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=26,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_27: variance - 1 tile
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=27,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_28: rstd - 1 tile
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=28,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_29: normalized output - Wt tiles
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=29,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_30: tilized gamma (persistent, only if gamma)
    if has_gamma:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=30,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_31: tilized beta (persistent, only if beta)
    if has_beta:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=31,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # ========== 4. COMPILE-TIME ARGS ==========
    # Pack epsilon as (bf16 << 16 | bf16)
    packed_eps = _float_to_bfloat16_packed(epsilon)

    # --- Reader CT args ---
    # [cb_input_rm, cb_gamma, cb_beta, cb_reduce_scaler, cb_eps,
    #  stick_size, Wt, has_gamma, has_beta,
    #  TensorAccessorArgs(input), TensorAccessorArgs(gamma)?, TensorAccessorArgs(beta)?]
    reader_ct_args = [
        0,  # cb_input_rm (c_0)
        2,  # cb_gamma (c_2)
        3,  # cb_beta (c_3)
        8,  # cb_reduce_scaler (c_8)
        9,  # cb_eps (c_9)
        stick_size,  # W * 2 bytes
        Wt,  # tiles per row
        has_gamma,
        has_beta,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.extend([0, 0])  # placeholder for absent gamma (TensorAccessorArgs uses 2 CT args)
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    else:
        reader_ct_args.extend([0, 0])  # placeholder for absent beta (TensorAccessorArgs uses 2 CT args)

    # --- Compute CT args (per core group - num_tile_rows differs) ---
    # [Wt, num_tile_rows, has_gamma, has_beta]
    # We handle two core groups with different tile_row counts via defines
    # Since CT args are per-core-range, we need separate kernel descriptors per group
    # if they differ. However, for simplicity and because the kernel-writer will use
    # runtime args for num_tile_rows, we pass the max and let runtime args control per-core.
    # Actually, the design says num_tile_rows is a CT arg. This means we need separate
    # kernel descriptors for each core group if they have different tile_row counts.
    # For now, we'll use runtime args for num_tile_rows to avoid complexity.
    # UPDATE: The design specifies num_tile_rows as CT arg index 1 for compute.
    # Since CT args must be the same across all cores for a kernel, we'll pass max and
    # also send num_tile_rows as runtime arg. The kernel will use the runtime arg.
    # To keep things simpler: make num_tile_rows a runtime arg for compute too.

    compute_ct_args = [
        Wt,  # tiles per row
        has_gamma,
        has_beta,
    ]

    # --- Writer CT args ---
    # [cb_out_rm, stick_size, Wt, TensorAccessorArgs(output)]
    writer_ct_args = [
        16,  # cb_out_rm (c_16)
        stick_size,
        Wt,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ========== 5. RUNTIME ARGS ==========
    reader_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    # Track current start page (RM sticks) across cores
    current_start_page = 0

    grid_w = compute_grid.x
    grid_h = compute_grid.y

    # Build a set of active core coordinates from the core groups
    def _iter_core_range_set(crs):
        """Yield (x, y) for each core in a CoreRangeSet."""
        for cr in crs.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    yield (x, y)

    # Map each active core to its tile-row count
    core_tile_rows = {}
    for x, y in _iter_core_range_set(core_group_1):
        core_tile_rows[(x, y)] = tile_rows_per_core_g1

    if tile_rows_per_core_g2 > 0:
        for x, y in _iter_core_range_set(core_group_2):
            core_tile_rows[(x, y)] = tile_rows_per_core_g2

    # Assign runtime args in row-major core order
    for y in range(grid_h):
        for x in range(grid_w):
            if (x, y) in core_tile_rows:
                num_tile_rows = core_tile_rows[(x, y)]
                # Each tile-row = 32 RM sticks. start_page_id is in sticks.
                start_page_id = current_start_page

                # Reader RT args: [input_addr, num_tile_rows, start_page_id, packed_eps, gamma_addr, beta_addr]
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),
                    num_tile_rows,
                    start_page_id,
                    packed_eps,
                    gamma_addr,
                    beta_addr,
                ]

                # Compute RT args: [num_tile_rows]
                compute_rt_args[x][y] = [num_tile_rows]

                # Writer RT args: [output_addr, num_tile_rows, start_page_id]
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),
                    num_tile_rows,
                    start_page_id,
                ]

                # Advance by num_tile_rows * 32 sticks
                current_start_page += num_tile_rows * 32
            else:
                # Idle core: MUST set empty args
                reader_rt_args[x][y] = []
                compute_rt_args[x][y] = []
                writer_rt_args[x][y] = []

    # ========== 6. KERNEL DESCRIPTORS ==========
    # Compute defines for optional gamma/beta
    defines = []
    if has_gamma:
        defines.append(("HAS_GAMMA", "1"))
    if has_beta:
        defines.append(("HAS_BETA", "1"))

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        defines=defines,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        defines=defines,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 7. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )
