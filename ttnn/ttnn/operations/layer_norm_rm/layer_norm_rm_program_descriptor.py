# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for row-major layer normalization.

Work unit: tile-row (32 sticks x full width = Wt tiles).
Grid: 1D linear across cores.
"""

import struct
from pathlib import Path

import ttnn

# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_packed_bf16(val: float) -> int:
    """
    Convert a float to packed bf16 format: (bf16 << 16 | bf16).
    This fills both halves of a uint32 with the same bf16 value.
    """
    # Pack as float32, extract bytes, truncate to bf16 (top 16 bits)
    f32_bytes = struct.pack(">f", val)
    bf16_val = int.from_bytes(f32_bytes[:2], byteorder="big")
    # Pack bf16 into both halves of uint32
    return (bf16_val << 16) | bf16_val


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor: Input tensor (bfloat16, ROW_MAJOR_LAYOUT, on device).
        output_tensor: Pre-allocated output tensor (same spec as input).
        gamma: Optional scale tensor (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT.
        beta: Optional shift tensor (1,1,1,W), bfloat16, ROW_MAJOR_LAYOUT.
        epsilon: Numerical stability constant.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op.
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)
    W = shape[-1]
    H = shape[-2]

    # Compute total height across all batch dims: product of all dims except W
    total_H = 1
    for i in range(rank - 1):
        total_H *= shape[i]

    Wt = W // 32  # Width in tiles
    num_tile_rows = total_H // 32  # Total tile-rows (each is 32 sticks)

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # Page sizes
    # For RM tensors, page = stick = W * element_size
    stick_size = W * input_tensor.element_size()  # W * 2 for bfloat16

    # Tile size for bfloat16 32x32 tiles
    tile_size = ttnn.tile_size(input_tensor.dtype)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Use a single core for simplicity. The design supports multi-core via
    # split_blocks_for_tilize, but single-core is sufficient for correctness
    # validation. Multi-core can be added as an optimization.
    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()

    # Distribute tile-rows across cores (1D linear)
    # Use the device grid but cap to available work
    max_cores = min(grid_size.x * grid_size.y, num_tile_rows)
    if max_cores <= 1:
        # Single core handles everything
        num_cores = 1
        core = ttnn.CoreCoord(0, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        nblocks_per_core = num_tile_rows
        nblocks_per_core_cliff = 0
        num_cores_with_work = 1
    else:
        # Distribute tile-rows across cores linearly
        nblocks_per_core = num_tile_rows // max_cores
        remainder = num_tile_rows % max_cores
        if nblocks_per_core == 0:
            # Fewer tile-rows than cores: use only as many cores as tile-rows
            num_cores = num_tile_rows
            nblocks_per_core = 1
            nblocks_per_core_cliff = 0
            remainder = 0
        else:
            num_cores = max_cores
            nblocks_per_core_cliff = nblocks_per_core + remainder if remainder > 0 else 0

        num_cores_with_work = num_cores

        # Build core range set: linear 1D across the grid
        cores = []
        count = 0
        for y in range(grid_size.y):
            for x in range(grid_size.x):
                if count >= num_cores:
                    break
                cores.append(ttnn.CoreCoord(x, y))
                count += 1
            if count >= num_cores:
                break

        if len(cores) == 1:
            core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
        else:
            core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[-1])])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices from design:
    # c_0:  RM input staging for tilize (Wt pages, tile-sized)
    # c_1:  Tilized input tiles (Wt pages)
    # c_2:  Reduce scaler 1/W (1 page)
    # c_3:  Epsilon scalar (1 page)
    # c_4:  Gamma tilized (Wt pages, optional)
    # c_5:  Beta tilized (Wt pages, optional)
    # c_16: Final tiles before untilize (Wt pages)
    # c_17: Untilized RM output (Wt pages, tile-sized)
    # c_24: Row-wise mean (1 page)
    # c_25: Intermediate tiles (Wt pages)
    # c_26: Row-wise variance (1 page)
    # c_27: Inv_std (1 page)

    dtype = input_tensor.dtype  # bfloat16

    cb_descriptors = []

    # c_0: RM input staging for tilize - Wt pages, tile-sized
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: Tilized input tiles - Wt pages
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=1,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_2: Reduce scaler 1/W - 1 page (bf16)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=2,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_3: Epsilon scalar - 1 page (bf16)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=3,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_4: Gamma tilized (optional) - Wt pages
    if has_gamma:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=4,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_5: Beta tilized (optional) - Wt pages
    if has_beta:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=5,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_16: Final tiles before untilize - Wt pages
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=16,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_17: Untilized RM output - Wt pages, tile-sized
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=17,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_24: Row-wise mean - 1 page
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=24,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_25: Intermediate tiles - Wt pages
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=25,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_26: Row-wise variance - 1 page
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=26,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_27: Inv_std - 1 page
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=27,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Compile-time args ---

    # Reader compile-time args: stick_size, has_gamma, has_beta, TensorAccessorArgs(input)
    reader_ct_args = [stick_size, has_gamma, has_beta]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Compute compile-time args: Wt, num_tile_rows_this_core (will vary per core), has_gamma, has_beta
    # Since compile-time args are the same for all cores, use the max (nblocks_per_core)
    # and let runtime args specify actual count per core.
    compute_ct_args = [Wt, nblocks_per_core, has_gamma, has_beta]

    # Writer compile-time args: stick_size, Wt, TensorAccessorArgs(output)
    writer_ct_args = [stick_size, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # --- Runtime args ---
    # Pack scaler and epsilon as bf16 packed values
    scaler_value = _float_to_packed_bf16(1.0 / W)
    eps_value = _float_to_packed_bf16(epsilon)

    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0
    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    # Iterate over all cores in the grid and assign work
    core_idx = 0
    start_stick_id = 0
    for y in range(grid_size.y):
        for x in range(grid_size.x):
            if core_idx < num_cores_with_work:
                # Determine tile-rows for this core
                if nblocks_per_core_cliff > 0 and core_idx == num_cores_with_work - 1:
                    # Last core (cliff) gets remainder
                    this_core_tile_rows = nblocks_per_core_cliff
                else:
                    this_core_tile_rows = nblocks_per_core

                this_core_start_stick = start_stick_id

                reader_rt_args[x][y] = [
                    src_addr,  # 0: src_addr
                    this_core_tile_rows,  # 1: num_tile_rows
                    this_core_start_stick,  # 2: start_stick_id
                    gamma_addr,  # 3: gamma_addr
                    beta_addr,  # 4: beta_addr
                    scaler_value,  # 5: scaler_value (packed bf16)
                    eps_value,  # 6: eps_value (packed bf16)
                ]

                writer_rt_args[x][y] = [
                    dst_addr,  # 0: dst_addr
                    this_core_tile_rows,  # 1: num_tile_rows
                    this_core_start_stick,  # 2: start_stick_id
                ]

                # Each tile-row is 32 sticks
                start_stick_id += this_core_tile_rows * 32
            else:
                # Idle core: MUST set empty list
                reader_rt_args[x][y] = []
                writer_rt_args[x][y] = []

            core_idx += 1

    # --- Kernel descriptors ---
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )
