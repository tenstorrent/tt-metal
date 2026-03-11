# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernel descriptors, and
per-core runtime args for in-kernel tilize/untilize layer normalization.

Work unit = tile-row (32 rows x Wt tiles).
Work is distributed across cores using split_work_to_cores().
"""

from pathlib import Path
import ttnn

# Kernel files live in kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"

# CB index assignments (matching op_design.md)
CB_INPUT_RM = 0  # c_0: RM sticks from reader (Wt pages, tile-sized)
CB_TILIZED = 1  # c_1: Tilized input tiles (Wt pages)
CB_REDUCE_SCALER = 2  # c_2: Reduce scaler 1/W (1 page, bf16)
CB_EPS_SCALAR = 3  # c_3: Epsilon tile (1 page, bf16)
CB_GAMMA = 4  # c_4: Gamma tiles (Wt pages)
CB_BETA = 5  # c_5: Beta tiles (Wt pages)
CB_OUTPUT_TILES = 16  # c_16: Pre-untilize output tiles (Wt pages)
CB_OUTPUT_RM = 17  # c_17: Untilized RM output (Wt pages)
CB_MEAN = 24  # c_24: Row-reduced mean (1 page)
CB_CENTERED = 25  # c_25: Centered tiles x-mean (Wt pages)
CB_VAR = 26  # c_26: Row-reduced variance (1 page)
CB_RSQRT_VAR = 27  # c_27: rsqrt(var+eps) (1 page)
CB_NORMALIZED = 28  # c_28: Normalized output tiles (Wt pages)


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
        input_tensor: Input tensor (bfloat16, RM, interleaved, on device).
        output_tensor: Pre-allocated output tensor (same spec).
        gamma: Optional scale tensor (1,1,1,W), bfloat16 RM, on device.
        beta: Optional shift tensor (1,1,1,W), bfloat16 RM, on device.
        epsilon: Variance stabilizer.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op().
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    ndim = len(shape)
    H = shape[-2]
    W = shape[-1]
    Wt = W // 32  # tiles per row
    TILE_H = 32

    # Batch*Channel dims (all dims except last two)
    NC = 1
    for i in range(ndim - 2):
        NC *= shape[i]

    Ht = H // TILE_H  # tile-rows per NC slice
    total_tile_rows = NC * Ht  # total work units

    # Page sizes
    tile_size = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes for 32x32 bf16
    stick_size = W * input_tensor.element_size()  # W * 2 bytes

    has_gamma = gamma is not None
    has_beta = beta is not None

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (num_cores, core_grid, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2) = ttnn.split_work_to_cores(
        all_cores, total_tile_rows
    )

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    cbs = []

    # c_0: RM input sticks -- tile-sized pages (tilize convention)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_RM,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: Tilized tiles
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

    # c_2: Reduce scaler (1/W), 1 tile, bf16
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_REDUCE_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_3: Epsilon scalar, 1 tile, bf16
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EPS_SCALAR,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_4: Gamma tiles (Wt pages) -- only needed if gamma present
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

    # c_5: Beta tiles (Wt pages) -- only needed if beta present
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

    # c_16: Output tiles (pre-untilize)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_17: Output RM (post-untilize)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_RM,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_24: Mean (1 tile, col vector)
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

    # c_25: Centered tiles (Wt pages)
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

    # c_26: Variance (1 tile, col vector)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_VAR,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_27: rsqrt(var+eps) (1 tile, col vector)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_RSQRT_VAR,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_28: Normalized tiles (Wt pages)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_NORMALIZED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    _populate_reader_runtime_args(
        reader_rt_args,
        input_tensor,
        gamma,
        beta,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
        compute_grid,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [Wt, int(has_gamma), int(has_beta)]

    compute_rt_args = ttnn.RuntimeArgs()
    _populate_compute_runtime_args(
        compute_rt_args,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
        compute_grid,
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [CB_OUTPUT_RM, stick_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    _populate_writer_runtime_args(
        writer_rt_args,
        output_tensor,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
        compute_grid,
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
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )


# ============================================================================
# Runtime args population helpers
# ============================================================================


def _iter_cores_with_work(core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2):
    """Yield (CoreCoord, nblocks) for every active core."""
    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                yield ttnn.CoreCoord(x, y), rows_per_core_g1
    for cr in core_group_2.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                yield ttnn.CoreCoord(x, y), rows_per_core_g2


def _populate_reader_runtime_args(
    rt_args,
    input_tensor,
    gamma,
    beta,
    core_group_1,
    core_group_2,
    rows_per_core_g1,
    rows_per_core_g2,
    compute_grid,
):
    """Set per-core runtime args for the reader kernel.

    Per design:
      [0] input_addr
      [1] start_stick_id (first stick for this core)
      [2] num_sticks (nblocks * 32)
      [3] gamma_addr (0 if none)
      [4] beta_addr (0 if none)
    """
    input_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    current_stick = 0
    active_cores = set()

    for core, nblocks in _iter_cores_with_work(core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2):
        num_sticks = nblocks * 32
        rt_args[core.x][core.y] = [
            input_addr,
            current_stick,
            num_sticks,
            gamma_addr,
            beta_addr,
        ]
        current_stick += num_sticks
        active_cores.add((core.x, core.y))

    # Set empty args for idle cores
    for x in range(compute_grid.x):
        for y in range(compute_grid.y):
            if (x, y) not in active_cores:
                rt_args[x][y] = []


def _populate_compute_runtime_args(
    rt_args,
    core_group_1,
    core_group_2,
    rows_per_core_g1,
    rows_per_core_g2,
    compute_grid,
):
    """Set per-core runtime args for the compute kernel.

    Per design, nblocks is a runtime arg so different cores can process
    different numbers of tile-rows:
      [0] nblocks (tile-rows for this core)
    """
    active_cores = set()

    for core, nblocks in _iter_cores_with_work(core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2):
        rt_args[core.x][core.y] = [nblocks]
        active_cores.add((core.x, core.y))

    for x in range(compute_grid.x):
        for y in range(compute_grid.y):
            if (x, y) not in active_cores:
                rt_args[x][y] = []


def _populate_writer_runtime_args(
    rt_args,
    output_tensor,
    core_group_1,
    core_group_2,
    rows_per_core_g1,
    rows_per_core_g2,
    compute_grid,
):
    """Set per-core runtime args for the writer kernel.

    Per design:
      [0] output_addr
      [1] start_stick_id
      [2] num_sticks
    """
    output_addr = output_tensor.buffer_address()

    current_stick = 0
    active_cores = set()

    for core, nblocks in _iter_cores_with_work(core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2):
        num_sticks = nblocks * 32
        rt_args[core.x][core.y] = [
            output_addr,
            current_stick,
            num_sticks,
        ]
        current_stick += num_sticks
        active_cores.add((core.x, core.y))

    for x in range(compute_grid.x):
        for y in range(compute_grid.y):
            if (x, y) not in active_cores:
                rt_args[x][y] = []
