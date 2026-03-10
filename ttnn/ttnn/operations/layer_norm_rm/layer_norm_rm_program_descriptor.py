# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Program Descriptor

Configures circular buffers, work distribution, and kernel descriptors for
row-major interleaved layer normalization.

Data flow: RM sticks -> tilize -> compute phases -> untilize -> RM sticks

CB Layout:
    c_0  (CB 0):  RM input sticks staging       [Wt pages, tile-sized]
    c_1  (CB 1):  Gamma tiles (optional)         [Wt pages, bf16 tile]
    c_2  (CB 2):  Beta tiles (optional)          [Wt pages, bf16 tile]
    c_8  (CB 8):  Reduce scaler (1/W)            [1 page, bf16 tile]
    c_10 (CB 10): Epsilon scalar tile            [1 page, bf16 tile]
    c_16 (CB 16): Untilized output               [Wt pages, bf16 tile]
    c_24 (CB 24): Tilized input                  [Wt pages, bf16 tile]
    c_25 (CB 25): Mean col vector                [2 pages, bf16 tile]
    c_26 (CB 26): Centered (x - mean)            [Wt pages, bf16 tile]
    c_27 (CB 27): Squared centered               [Wt pages, bf16 tile]
    c_28 (CB 28): Variance col vector            [2 pages, fp32 tile]
    c_29 (CB 29): Inverse std (rsqrt(var+eps))   [2 pages, bf16 tile]
    c_30 (CB 30): Pre-untilize final tiles       [Wt pages, bf16 tile]
"""

from pathlib import Path
import ttnn

# Kernel files are in the kernels/ subdirectory, relative to tt-metal root
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create ProgramDescriptor for layer_norm_rm.

    Work unit: tile-row (one horizontal row of Wt tiles = 32 RM sticks).
    Grid: 1D distribution of tile-rows across cores.
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    rank = len(shape)
    W = shape[rank - 1]
    Wt = W // 32  # tiles per row

    # Flatten all dims except last to get total height
    H_total = 1
    for i in range(rank - 1):
        H_total *= shape[i]
    nblocks = H_total // 32  # number of tile-rows (each = 32 RM sticks)

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # Page sizes
    # RM stick size for input/output (row-major): W elements * 2 bytes (bf16)
    stick_size = W * 2  # bytes per RM stick

    # Tile page size for bf16: standard 32x32 tile = 2048 bytes
    # NOTE: input_tensor.buffer_page_size() returns stick_size for RM tensors,
    # NOT tile_size. We must use ttnn.tile_size() for CB page sizes.
    bf16_tile_size = ttnn.tile_size(ttnn.bfloat16)  # 2048

    # fp32 tile size for variance accumulation (c_28)
    fp32_tile_size = ttnn.tile_size(ttnn.float32)  # 4096

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    # Use single-core for simplicity to ensure correctness first.
    # split_work_to_cores can be used for multi-core later.
    # For stub kernels, single core is safest.
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    num_rows_per_core = nblocks

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    cbs = []

    # c_0: RM input sticks staging (Wt pages, tile-sized each)
    # Reader writes RM sticks here, compute tilizes from here
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=0,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_1: Gamma tiles (optional, Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=1,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_2: Beta tiles (optional, Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=2,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_8: Reduce scaler (1/W), 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=8,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_10: Epsilon scalar tile, 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=1 * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=10,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_16: Untilized output (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=16,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_24: Tilized input (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=24,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_25: Mean col vector (2 pages for double-buffer)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=25,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_26: Centered x - mean (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=26,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_27: Squared centered (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=27,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_28: Variance col vector (2 pages, fp32 for accumulation precision)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * fp32_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=28,
                    data_format=ttnn.float32,
                    page_size=fp32_tile_size,
                )
            ],
        )
    )

    # c_29: Inverse std rsqrt(var + eps) (2 pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=29,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # c_30: Pre-untilize final tiles (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * bf16_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=30,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [
        stick_size,  # arg 0: RM stick size in bytes
        Wt,  # arg 1: tiles per row
        has_gamma,  # arg 2: 1 if gamma provided
        has_beta,  # arg 3: 1 if beta provided
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),  # arg 0: input DRAM address
        num_rows_per_core,  # arg 1: tile-rows on this core
        0,  # arg 2: start_stick_id (first RM stick)
        gamma_addr,  # arg 3: gamma DRAM address (0 if none)
        beta_addr,  # arg 4: beta DRAM address (0 if none)
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    writer_ct_args = [
        stick_size,  # arg 0: RM stick size in bytes
        Wt,  # arg 1: tiles per row
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),  # arg 0: output DRAM address
        num_rows_per_core,  # arg 1: tile-rows on this core
        0,  # arg 2: start_stick_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [
        num_rows_per_core,  # arg 0: tile-rows to process
        Wt,  # arg 1: tiles per row
        has_gamma,  # arg 2: 1 if gamma provided
        has_beta,  # arg 3: 1 if beta provided
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
