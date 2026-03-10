# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for layer normalization on row-major interleaved tensors.

Data flow per block (1 tile-row = 32 sticks spanning full width = Wt tiles):
    RM sticks (DRAM) -> [reader] -> c_0 -> [compute: tilize] -> c_16
    -> [compute: reduce_mean] -> c_24 -> [compute: sub_mean] -> c_25
    -> [compute: square] -> c_16 -> [compute: reduce_var] -> c_24
    -> [compute: eps+rsqrt] -> c_27 -> [compute: mul_rstd] -> c_16
    -> [optional: mul_gamma] -> c_25 -> [optional: add_beta] -> c_16
    -> [compute: untilize] -> c_17 -> [writer] -> RM sticks (DRAM)

CB Layout:
    c_0  (CB 0):  RM sticks from reader         (Wt pages)
    c_5  (CB 5):  Tilized gamma (optional)       (Wt pages, program lifetime)
    c_6  (CB 6):  Tilized beta (optional)        (Wt pages, program lifetime)
    c_8  (CB 8):  Reduce scaler (1/W)            (1 page, program lifetime)
    c_9  (CB 9):  Epsilon constant tile           (1 page, program lifetime)
    c_16 (CB 16): Multi-use: tilized input, squared, normalized, pre-untilize (Wt pages)
    c_17 (CB 17): Untilized output for writer    (Wt pages)
    c_24 (CB 24): Reduce output (mean/variance)  (1 page)
    c_25 (CB 25): Centered values / affine intermediates (Wt pages)
    c_27 (CB 27): rsqrt(var+eps)                 (1 page)
"""

from pathlib import Path
import ttnn

# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


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
        input_tensor: Input tensor (bfloat16, ROW_MAJOR, interleaved, on device)
        output_tensor: Pre-allocated output tensor (same specs as input)
        gamma: Optional scale parameter tensor (1,1,1,W)
        beta: Optional shift parameter tensor (1,1,1,W)
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)

    W = shape[-1]  # Width (last dim)
    H = shape[-2]  # Height (second-to-last dim)
    # Flatten all dims except W and H into total height
    H_total = 1
    for i in range(rank - 2):
        H_total *= shape[i]
    H_total *= H

    Wt = W // 32  # Tiles per row (width dimension)

    # tile_size for bfloat16 32x32 tiles
    tile_size = ttnn.tile_size(input_tensor.dtype)

    # RM stick size = W * element_size (2 bytes for bfloat16)
    stick_size = W * input_tensor.element_size()

    # Total blocks = total height / 32 (each block is 32 sticks = 1 tile-row)
    total_blocks = H_total // 32

    # ========== 2. WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(compute_grid, total_blocks)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB page sizes:
    #   For tile CBs: tile_size (bfloat16 32x32 tile)
    #   For RM CBs (c_0, c_17): also tile_size, since RM sticks are packed into tile-sized pages
    #     (each tile-page holds 32 RM sticks for a given tile-column)
    # Note: the reader writes 32 sticks into Wt tile-sized pages in c_0.

    # RM page size for c_0 and c_17 is tile_size (same physical size, different interpretation)
    rm_page_size = tile_size

    # CB 0: Input RM sticks (Wt pages per block)
    cb_0 = ttnn.CBDescriptor(
        total_size=Wt * rm_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=input_tensor.dtype,
                page_size=rm_page_size,
            )
        ],
    )

    # CB 5: Tilized gamma (optional, Wt pages, program lifetime)
    cb_5 = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=5,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 6: Tilized beta (optional, Wt pages, program lifetime)
    cb_6 = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=6,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 8: Reduce scaler (1/W), 1 tile, bfloat16
    cb_8 = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=8,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 9: Epsilon constant tile, 1 tile, bfloat16
    cb_9 = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=9,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 16: Multi-use tilized buffer (Wt pages, reused per phase)
    cb_16 = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=16,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 17: Untilized output for writer (Wt pages per block)
    cb_17 = ttnn.CBDescriptor(
        total_size=Wt * rm_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=17,
                data_format=output_tensor.dtype,
                page_size=rm_page_size,
            )
        ],
    )

    # CB 24: Reduce output (mean/variance), 1 tile
    cb_24 = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=24,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 25: Centered values / affine intermediates (Wt pages)
    cb_25 = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=25,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # CB 27: rsqrt(var+eps), 1 tile
    cb_27 = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=27,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cbs = [cb_0, cb_5, cb_6, cb_8, cb_9, cb_16, cb_17, cb_24, cb_25, cb_27]

    # ========== 4. KERNEL DESCRIPTORS ==========
    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # --- Reader kernel ---
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = _build_reader_runtime_args(
        input_tensor,
        gamma,
        beta,
        epsilon,
        all_cores,
        compute_grid,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Compile-time args differ per core group (different num_blocks_per_core)
    # We use core_group_1 and core_group_2 with separate kernel descriptors
    compute_kernels = _build_compute_kernels(
        Wt,
        has_gamma,
        has_beta,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    )

    # --- Writer kernel ---
    writer_ct_args = [stick_size, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = _build_writer_runtime_args(
        output_tensor,
        all_cores,
        compute_grid,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    kernels = [reader_kernel, writer_kernel] + compute_kernels

    return ttnn.ProgramDescriptor(
        kernels=kernels,
        cbs=cbs,
        semaphores=[],
    )


def _build_compute_kernels(
    Wt,
    has_gamma,
    has_beta,
    core_group_1,
    core_group_2,
    blocks_per_core_g1,
    blocks_per_core_g2,
):
    """Build compute kernel descriptors for each core group."""
    kernels = []
    kernel_source = str(KERNEL_DIR / "layer_norm_rm_compute.cpp")

    if blocks_per_core_g1 > 0:
        ct_args_g1 = [blocks_per_core_g1, Wt, has_gamma, has_beta]
        compute_rt_args_g1 = ttnn.RuntimeArgs()  # Compute has no runtime args
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kernel_source,
                core_ranges=core_group_1,
                compile_time_args=ct_args_g1,
                runtime_args=compute_rt_args_g1,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=False,
                    math_approx_mode=False,
                ),
            )
        )

    if blocks_per_core_g2 > 0:
        ct_args_g2 = [blocks_per_core_g2, Wt, has_gamma, has_beta]
        compute_rt_args_g2 = ttnn.RuntimeArgs()
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kernel_source,
                core_ranges=core_group_2,
                compile_time_args=ct_args_g2,
                runtime_args=compute_rt_args_g2,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=False,
                    math_approx_mode=False,
                ),
            )
        )

    return kernels


def _build_reader_runtime_args(
    input_tensor,
    gamma,
    beta,
    epsilon,
    all_cores,
    compute_grid,
    core_group_1,
    core_group_2,
    blocks_per_core_g1,
    blocks_per_core_g2,
):
    """Build per-core runtime args for the reader kernel."""
    import struct

    rt_args = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0
    # Bit-cast epsilon float to uint32_t for passing as runtime arg
    eps_bits = struct.unpack("I", struct.pack("f", epsilon))[0]

    # Iterate over all cores in grid, assign work
    start_stick_id = 0
    grid_w = compute_grid.x
    grid_h = compute_grid.y

    # Track which cores are active via core groups
    g1_cores = _core_range_set_to_set(core_group_1)
    g2_cores = _core_range_set_to_set(core_group_2)

    for y in range(grid_h):
        for x in range(grid_w):
            core_coord = (x, y)
            if core_coord in g1_cores:
                num_blocks = blocks_per_core_g1
            elif core_coord in g2_cores:
                num_blocks = blocks_per_core_g2
            else:
                # Idle core
                rt_args[x][y] = []
                continue

            num_sticks = num_blocks * 32
            rt_args[x][y] = [
                src_addr,
                start_stick_id,
                num_sticks,
                gamma_addr,
                beta_addr,
                eps_bits,
            ]
            start_stick_id += num_sticks

    return rt_args


def _build_writer_runtime_args(
    output_tensor,
    all_cores,
    compute_grid,
    core_group_1,
    core_group_2,
    blocks_per_core_g1,
    blocks_per_core_g2,
):
    """Build per-core runtime args for the writer kernel."""
    rt_args = ttnn.RuntimeArgs()

    dst_addr = output_tensor.buffer_address()

    start_stick_id = 0
    grid_w = compute_grid.x
    grid_h = compute_grid.y

    g1_cores = _core_range_set_to_set(core_group_1)
    g2_cores = _core_range_set_to_set(core_group_2)

    for y in range(grid_h):
        for x in range(grid_w):
            core_coord = (x, y)
            if core_coord in g1_cores:
                num_blocks = blocks_per_core_g1
            elif core_coord in g2_cores:
                num_blocks = blocks_per_core_g2
            else:
                rt_args[x][y] = []
                continue

            num_sticks = num_blocks * 32
            rt_args[x][y] = [
                dst_addr,
                start_stick_id,
                num_sticks,
            ]
            start_stick_id += num_sticks

    return rt_args


def _core_range_set_to_set(core_range_set):
    """Convert a CoreRangeSet to a Python set of (x, y) tuples for fast lookup."""
    result = set()
    for cr in core_range_set.ranges():
        x_start = cr.start.x
        y_start = cr.start.y
        x_end = cr.end.x
        y_end = cr.end.y
        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                result.add((x, y))
    return result
