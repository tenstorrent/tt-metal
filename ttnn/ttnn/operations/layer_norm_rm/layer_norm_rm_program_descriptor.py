# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.

Work unit = tile-row block (32 RM sticks x full width = Wt tiles).
total_blocks = N * C * H / 32.

CB layout (from op_design.md):
  c_0  (Wt pages): RM input sticks from DRAM (tile-sized pages)
  c_1  (Wt tiles): tilized gamma (optional, program lifetime)
  c_2  (Wt tiles): tilized beta (optional, program lifetime)
  c_8  (1 tile):   reduce scaler 1/W (bf16, program lifetime)
  c_9  (1 tile):   epsilon constant (bf16, program lifetime)
  c_16 (Wt pages): untilized RM output (tile-sized pages)
  c_24 (Wt tiles): tilized input / reused intermediate
  c_25 (1 tile):   row mean / reused for variance
  c_26 (Wt tiles): centered values (x - mean)
  c_27 (Wt tiles): squared centered / reused for affine output
  c_28 (1 tile):   inv_std = rsqrt(var + eps)
"""

import struct
from pathlib import Path

import ttnn

# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices
CB_RM_INPUT = 0  # RM sticks from DRAM
CB_GAMMA = 1  # Tilized gamma (optional)
CB_BETA = 2  # Tilized beta (optional)
CB_SCALER = 8  # Reduce scaler (1/W)
CB_EPS = 9  # Epsilon constant tile
CB_RM_OUTPUT = 16  # Untilized RM output
CB_TILIZED = 24  # Tilized input / intermediate
CB_MEAN = 25  # Row mean / reused for var
CB_CENTERED = 26  # x - mean
CB_SQUARED = 27  # (x-mean)^2 / reused for affine
CB_INV_STD = 28  # rsqrt(var + eps)


def _float_to_uint32(f: float) -> int:
    """Convert a float to its uint32 bit representation."""
    return struct.unpack("I", struct.pack("f", f))[0]


def _bf16_to_uint16(f: float) -> int:
    """Convert a float to bfloat16 uint16 representation (truncate lower 16 bits of float32)."""
    bits = struct.unpack("I", struct.pack("f", f))[0]
    return (bits >> 16) & 0xFFFF


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
        input_tensor: RM interleaved bfloat16 input on device
        output_tensor: Pre-allocated RM interleaved bfloat16 output on device
        gamma: Optional gamma tensor on device (1,1,1,W), RM bf16
        beta: Optional beta tensor on device (1,1,1,W), RM bf16
        epsilon: Stability constant

    Returns:
        ProgramDescriptor ready for ttnn.generic_op
    """
    device = input_tensor.device()
    has_gamma = gamma is not None
    has_beta = beta is not None

    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)
    TILE_H, TILE_W = 32, 32
    H = shape[rank - 2]
    W = shape[rank - 1]
    Wt = W // TILE_W  # tiles per row

    # Compute NC (product of all dims except last two)
    NC = 1
    for i in range(rank - 2):
        NC *= shape[i]

    # total_blocks = number of tile-row blocks
    total_blocks = NC * (H // TILE_H)

    # Page sizes
    tile_size = ttnn.tile_size(ttnn.bfloat16)  # 32*32*2 = 2048 bytes
    # RM stick size for the input tensor
    stick_size = W * input_tensor.element_size()  # W * 2 bytes

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    compute_grid = device.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (
        num_cores,
        core_grid,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(all_cores, total_blocks)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    cbs = []

    # c_0: RM input (Wt tile-sized pages per block)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RM_INPUT,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: gamma (Wt tiles, optional but always allocated for simplicity)
    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_2: beta (Wt tiles, optional)
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_8: reduce scaler (1 tile, bf16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_9: epsilon constant (1 tile, bf16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EPS,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_16: RM output (Wt tile-sized pages per block)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RM_OUTPUT,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_24: tilized input / reused intermediate (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_TILIZED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_25: mean / variance (1 tile)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MEAN,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_26: centered values (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_27: squared / affine intermediate (Wt tiles)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SQUARED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_28: inv_std (1 tile)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INV_STD,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [stick_size]  # index 0: stick_size = W * 2
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.append(1 if has_gamma else 0)  # has_gamma
    reader_ct_args.append(1 if has_beta else 0)  # has_beta
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    # Pack epsilon as bfloat16 doubled: (bf16 << 16) | bf16
    eps_bf16 = _bf16_to_uint16(epsilon)
    eps_packed = (eps_bf16 << 16) | eps_bf16

    reader_rt_args = ttnn.RuntimeArgs()
    current_block = 0

    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                start_stick = current_block * TILE_H
                reader_rt_args[x][y] = [
                    input_tensor.buffer_address(),  # src_addr
                    blocks_per_core_g1,  # num_blocks
                    start_stick,  # start_stick_id
                    Wt,  # Wt (tiles per row)
                    W,  # W (width in elements)
                    gamma.buffer_address() if has_gamma else 0,  # gamma_addr
                    beta.buffer_address() if has_beta else 0,  # beta_addr
                    eps_packed,  # eps_packed (bf16 doubled)
                ]
                current_block += blocks_per_core_g1

    if blocks_per_core_g2 > 0:
        for cr in core_group_2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    start_stick = current_block * TILE_H
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        blocks_per_core_g2,
                        start_stick,
                        Wt,
                        W,
                        gamma.buffer_address() if has_gamma else 0,
                        beta.buffer_address() if has_beta else 0,
                        eps_packed,  # eps_packed (bf16 doubled)
                    ]
                    current_block += blocks_per_core_g2

    # Fill empty runtime args for idle cores
    for xi in range(compute_grid.x):
        for yi in range(compute_grid.y):
            try:
                _ = reader_rt_args[xi][yi]
            except Exception:
                reader_rt_args[xi][yi] = []

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Compile-time: Wt, max_num_blocks, has_gamma, has_beta
    max_blocks = max(blocks_per_core_g1, blocks_per_core_g2) if blocks_per_core_g2 > 0 else blocks_per_core_g1
    compute_ct_args = [
        Wt,
        max_blocks,
        1 if has_gamma else 0,
        1 if has_beta else 0,
    ]

    compute_rt_args = ttnn.RuntimeArgs()
    current_block = 0

    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                compute_rt_args[x][y] = [blocks_per_core_g1]
                current_block += blocks_per_core_g1

    if blocks_per_core_g2 > 0:
        for cr in core_group_2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    compute_rt_args[x][y] = [blocks_per_core_g2]
                    current_block += blocks_per_core_g2

    for xi in range(compute_grid.x):
        for yi in range(compute_grid.y):
            try:
                _ = compute_rt_args[xi][yi]
            except Exception:
                compute_rt_args[xi][yi] = []

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [stick_size]  # index 0: stick_size
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    current_block = 0

    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                start_stick = current_block * TILE_H
                writer_rt_args[x][y] = [
                    output_tensor.buffer_address(),  # dst_addr
                    blocks_per_core_g1,  # num_blocks
                    start_stick,  # start_stick_id
                    Wt,  # Wt
                ]
                current_block += blocks_per_core_g1

    if blocks_per_core_g2 > 0:
        for cr in core_group_2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    start_stick = current_block * TILE_H
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        blocks_per_core_g2,
                        start_stick,
                        Wt,
                    ]
                    current_block += blocks_per_core_g2

    for xi in range(compute_grid.x):
        for yi in range(compute_grid.y):
            try:
                _ = writer_rt_args[xi][yi]
            except Exception:
                writer_rt_args[xi][yi] = []

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
