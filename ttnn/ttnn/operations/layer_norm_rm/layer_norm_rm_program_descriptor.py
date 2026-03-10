# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines ProgramDescriptor: circular buffers, kernels, runtime args, and work distribution
for row-wise layer normalization on ROW_MAJOR interleaved tensors.

CB Layout (from op_design.md):
  c_0  : input RM sticks (tile-sized pages) - Reader -> Compute(tilize)
  c_1  : tilized input tiles               - Compute(tilize) -> Compute(sub,square)
  c_2  : reduce scaler tile (1/W)          - Reader -> Compute(reduce)
  c_3  : mean per row (reduced)            - Compute(reduce) -> Compute(sub)
  c_4  : centered tiles (x - mean)         - Compute(sub) -> Compute(square,mul)
  c_5  : squared centered tiles            - Compute(square) -> Compute(reduce_var)
  c_6  : inv_std = rsqrt(var + eps)        - Compute(reduce+rsqrt) -> Compute(mul)
  c_7  : epsilon tile (constant)           - Reader -> Compute(add_eps)
  c_16 : output RM sticks (tile-sized)     - Compute(untilize) -> Writer
  c_24 : normalized tiles (before affine)  - Compute(mul) -> Compute(affine or untilize)
  c_25 : tilized gamma tiles               - Compute(tilize) -> Compute(mul_gamma)
  c_26 : tilized beta tiles                - Compute(tilize) -> Compute(add_beta)
  c_27 : gamma RM sticks (tile-sized)      - Reader -> Compute(tilize_gamma)
  c_28 : beta RM sticks (tile-sized)       - Reader -> Compute(tilize_beta)
  c_29 : output after affine transform     - Compute(affine) -> Compute(untilize)
"""

from pathlib import Path
import struct
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
        input_tensor: Input tensor (bfloat16, ROW_MAJOR, on device)
        output_tensor: Pre-allocated output tensor (bfloat16, ROW_MAJOR, on device)
        gamma: Optional gamma tensor (1,1,1,W) (bfloat16, ROW_MAJOR)
        beta: Optional beta tensor (1,1,1,W) (bfloat16, ROW_MAJOR)
        epsilon: Stability constant for variance

    Returns:
        ProgramDescriptor ready for ttnn.generic_op
    """
    has_gamma = gamma is not None
    has_beta = beta is not None

    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)
    W = shape[-1]
    H = shape[-2]

    # Compute batch dimensions (everything except last 2)
    batch_size = 1
    for i in range(rank - 2):
        batch_size *= shape[i]

    # Tile dimensions
    TILE_H = 32
    TILE_W = 32
    Wt = W // TILE_W  # tiles per row
    Ht = H // TILE_H  # tile-rows per sample

    # Total tile-row blocks (each block = 32 sticks spanning full width)
    total_blocks = batch_size * Ht

    # Page sizes
    # For ROW_MAJOR tensors, stick size = W * element_size
    input_stick_size = W * input_tensor.element_size()  # bytes per input stick (W * 2 for bfloat16)

    # Output dimensions (may differ from input, e.g., reduce_mean stage outputs W=32)
    output_W = output_tensor.shape[-1]
    output_Wt = output_W // TILE_W
    output_stick_size = output_W * output_tensor.element_size()

    # For tile-sized CBs, use tile_size from ttnn
    tile_size = ttnn.tile_size(input_tensor.dtype)  # 32x32 tile size in bytes

    # For RM CBs, page_size = tile_size (tile-sized pages for tilize/untilize compatibility)
    rm_page_size = tile_size

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    # Create full core range for work distribution
    max_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    all_cores_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (
        num_cores,
        core_grid,
        core_group_1,
        core_group_2,
        blocks_per_core_group_1,
        blocks_per_core_group_2,
    ) = ttnn.split_work_to_cores(all_cores_range, total_blocks)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    cbs = []

    # c_0: input RM sticks (tile-sized pages) - Wt pages per block
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * rm_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=input_tensor.dtype,
                    page_size=rm_page_size,
                )
            ],
        )
    )

    # c_1: tilized input tiles - Wt tiles per block
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=1,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_2: reduce scaler tile (1/W) - 1 tile, persistent
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=2,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_3: mean per row (reduced) - 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=3,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_4: centered tiles (x - mean) - Wt tiles
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=4,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_5: squared centered tiles - Wt tiles
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=5,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_6: inv_std = rsqrt(var + eps) - 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=6,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_7: epsilon tile (constant) - 1 tile, persistent
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=7,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_16: output RM sticks (tile-sized pages) - output_Wt pages per block
    cbs.append(
        ttnn.CBDescriptor(
            total_size=output_Wt * rm_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=16,
                    data_format=output_tensor.dtype,
                    page_size=rm_page_size,
                )
            ],
        )
    )

    # c_24: normalized tiles (before affine) - Wt tiles
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=24,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_25: tilized gamma tiles - Wt tiles (persistent, only if gamma)
    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=25,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_26: tilized beta tiles - Wt tiles (persistent, only if beta)
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=26,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_27: gamma RM sticks (tile-sized pages) - Wt pages (only if gamma)
    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * rm_page_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=27,
                        data_format=input_tensor.dtype,
                        page_size=rm_page_size,
                    )
                ],
            )
        )

    # c_28: beta RM sticks (tile-sized pages) - Wt pages (only if beta)
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * rm_page_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=28,
                        data_format=input_tensor.dtype,
                        page_size=rm_page_size,
                    )
                ],
            )
        )

    # c_29: output after affine transform - Wt tiles (only if gamma or beta)
    if has_gamma or has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=29,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [
        input_stick_size,  # 0: stick_size (bytes per input stick)
        1 if has_gamma else 0,  # 1: has_gamma
        1 if has_beta else 0,  # 2: has_beta
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    reader_rt_args = _build_reader_runtime_args(
        input_tensor,
        gamma,
        beta,
        epsilon,
        core_group_1,
        core_group_2,
        blocks_per_core_group_1,
        blocks_per_core_group_2,
        Wt,
        compute_grid,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    # nblocks_per_core is a compile-time arg and differs between core groups.
    # Create separate kernel descriptors for each core group.
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )

    compute_ct_args_1 = [
        blocks_per_core_group_1,  # 0: nblocks_per_core
        Wt,  # 1: Wt (tiles per row)
        1 if has_gamma else 0,  # 2: has_gamma
        1 if has_beta else 0,  # 3: has_beta
    ]

    compute_kernel_1 = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_group_1,
        compile_time_args=compute_ct_args_1,
        runtime_args=[],
        config=compute_config,
    )

    compute_kernels = [compute_kernel_1]

    if blocks_per_core_group_2 > 0:
        compute_ct_args_2 = [
            blocks_per_core_group_2,  # 0: nblocks_per_core (cliff core)
            Wt,  # 1: Wt (tiles per row)
            1 if has_gamma else 0,  # 2: has_gamma
            1 if has_beta else 0,  # 3: has_beta
        ]
        compute_kernel_2 = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
            core_ranges=core_group_2,
            compile_time_args=compute_ct_args_2,
            runtime_args=[],
            config=compute_config,
        )
        compute_kernels.append(compute_kernel_2)

    # --- Writer kernel ---
    writer_ct_args = [
        output_stick_size,  # 0: stick_size (bytes per output stick)
        output_Wt,  # 1: Wt (tiles per row for output)
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = _build_writer_runtime_args(
        output_tensor,
        core_group_1,
        core_group_2,
        blocks_per_core_group_1,
        blocks_per_core_group_2,
        compute_grid,
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel] + compute_kernels,
        cbs=cbs,
        semaphores=[],
    )


def _build_reader_runtime_args(
    input_tensor,
    gamma,
    beta,
    epsilon,
    core_group_1,
    core_group_2,
    blocks_per_core_group_1,
    blocks_per_core_group_2,
    Wt,
    compute_grid,
):
    """
    Build per-core runtime args for reader kernel.

    Runtime args per core:
      [0] src_addr        - Input buffer base address
      [1] num_sticks      - Total sticks for this core (nblocks * 32)
      [2] Wt              - Tiles per row
      [3] start_stick_id  - First stick ID for this core
      [4] gamma_addr      - Gamma buffer address (0 if none)
      [5] beta_addr       - Beta buffer address (0 if none)
      [6] eps_bits        - Epsilon as bit-cast uint32_t
    """
    rt_args = ttnn.RuntimeArgs()
    src_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0
    # Bit-cast epsilon float to uint32_t for passing as runtime arg
    eps_bits = struct.unpack("I", struct.pack("f", epsilon))[0]

    current_block = 0

    # Core group 1
    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                num_sticks = blocks_per_core_group_1 * 32
                start_stick_id = current_block * 32
                rt_args[x][y] = [src_addr, num_sticks, Wt, start_stick_id, gamma_addr, beta_addr, eps_bits]
                current_block += blocks_per_core_group_1

    # Core group 2
    for cr in core_group_2.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                num_sticks = blocks_per_core_group_2 * 32
                start_stick_id = current_block * 32
                rt_args[x][y] = [src_addr, num_sticks, Wt, start_stick_id, gamma_addr, beta_addr, eps_bits]
                current_block += blocks_per_core_group_2

    # Set empty args for idle cores
    _set_empty_for_idle_cores(rt_args, core_group_1, core_group_2, compute_grid)

    return rt_args


def _build_writer_runtime_args(
    output_tensor,
    core_group_1,
    core_group_2,
    blocks_per_core_group_1,
    blocks_per_core_group_2,
    compute_grid,
):
    """
    Build per-core runtime args for writer kernel.

    Runtime args per core:
      [0] dst_addr        - Output buffer base address
      [1] num_blocks      - Number of tile-row blocks for this core
      [2] start_stick_id  - First output stick ID
    """
    rt_args = ttnn.RuntimeArgs()
    dst_addr = output_tensor.buffer_address()

    current_block = 0

    # Core group 1
    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                start_stick_id = current_block * 32
                rt_args[x][y] = [dst_addr, blocks_per_core_group_1, start_stick_id]
                current_block += blocks_per_core_group_1

    # Core group 2
    for cr in core_group_2.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                start_stick_id = current_block * 32
                rt_args[x][y] = [dst_addr, blocks_per_core_group_2, start_stick_id]
                current_block += blocks_per_core_group_2

    # Set empty args for idle cores
    _set_empty_for_idle_cores(rt_args, core_group_1, core_group_2, compute_grid)

    return rt_args


def _set_empty_for_idle_cores(rt_args, core_group_1, core_group_2, compute_grid):
    """Set empty runtime args for cores not in any active group."""
    active_cores = set()

    for cg in [core_group_1, core_group_2]:
        for cr in cg.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    active_cores.add((x, y))

    for x in range(compute_grid.x):
        for y in range(compute_grid.y):
            if (x, y) not in active_cores:
                rt_args[x][y] = []
