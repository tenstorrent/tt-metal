# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.
Handles both dim=-1 (width reduction) and dim=-2 (height reduction) paths.
"""

import struct
from pathlib import Path
import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_uint32(f: float) -> int:
    """Bit-cast a float32 value to uint32."""
    return struct.unpack("I", struct.pack("f", f))[0]


def _pack_two_bfloat16(val: float) -> int:
    """Pack a float value as two bfloat16 values into a uint32.

    Format: (bf16 << 16 | bf16), where bf16 is the upper 16 bits of float32.
    """
    float_bits = struct.unpack("I", struct.pack("f", val))[0]
    bf16_bits = float_bits >> 16
    return (bf16_bits << 16) | bf16_bits


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for softmax.

    Args:
        input_tensor: Input tensor (on device, bfloat16, TILE_LAYOUT)
        output_tensor: Pre-allocated output tensor (on device)
        dim: Reduction dimension (-1 for width, -2 for height)
        numeric_stable: Whether to subtract max before exp

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    if dim == -1:
        return _create_program_descriptor_dim_w(input_tensor, output_tensor, numeric_stable)
    else:
        return _create_program_descriptor_dim_h(input_tensor, output_tensor, numeric_stable)


def _create_program_descriptor_dim_w(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    numeric_stable: bool,
) -> ttnn.ProgramDescriptor:
    """Program descriptor for softmax dim=-1 (width reduction)."""

    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    NC = N * C
    Ht = H // 32  # tile rows
    Wt = W // 32  # tile cols

    page_size = input_tensor.buffer_page_size()
    total_tiles = input_tensor.buffer_num_pages()

    # Work unit = one tile-row of Wt tiles
    total_work_units = NC * Ht

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    (num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2) = ttnn.split_work_to_cores(
        compute_grid, total_work_units
    )

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices from design:
    # c_0: input row tiles (Wt pages)
    # c_2: reduce scaler (1 page, persistent)
    # c_16: output (2 pages, double-buffered)
    # c_24: max per row (1 page, stable mode only)
    # c_25: exp values (Wt pages)
    # c_26: 1/sum(exp) per row (1 page)

    cb_input_id = 0
    cb_scaler_id = 2
    cb_out_id = 16
    cb_max_id = 24
    cb_exp_id = 25
    cb_recip_id = 26

    cbs = []

    # CB 0: input - Wt pages for one row
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_input_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 2: reduce scaler - 1 page (persistent, holds 1.0 in bfloat16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_scaler_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 16: output - 2 pages (double-buffered)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_out_id,
                    data_format=output_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 24: max tile - 1 page (for numeric_stable mode)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_max_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 25: exp values - Wt pages
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_exp_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 26: reciprocal of sum - 1 page
    cbs.append(
        ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_recip_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel (dim=-1) ---
    scaler_bits = _pack_two_bfloat16(1.0)
    reader_ct_args = [scaler_bits]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = _build_runtime_args_reader_w(
        input_tensor, compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2, Wt
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_reader_w.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel (dim=-1) ---
    # Compile-time args: Ht_per_unit=1 (processing one row at a time), Wt, NC=1 (batch folded), numeric_stable
    compute_ct_args = [1, Wt, 1, 1 if numeric_stable else 0]

    compute_rt_args = _build_runtime_args_compute(
        compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_compute_w.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [cb_out_id]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = _build_runtime_args_writer_w(
        output_tensor, compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2, Wt
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )


def _create_program_descriptor_dim_h(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    numeric_stable: bool,
) -> ttnn.ProgramDescriptor:
    """Program descriptor for softmax dim=-2 (height reduction)."""

    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    NC = N * C
    Ht = H // 32
    Wt = W // 32

    page_size = input_tensor.buffer_page_size()

    # Work unit = one tile-column of Ht tiles
    total_work_units = NC * Wt

    # chunk_size for dim=-2: process all Wt columns assigned to a core in one chunk
    # For simplicity, chunk_size = min(8, Wt) where 8 = DEST register limit for bf16 SyncHalf
    DEST_TILE_LIMIT = 8
    chunk_size = min(DEST_TILE_LIMIT, Wt)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    (num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2) = ttnn.split_work_to_cores(
        compute_grid, total_work_units
    )

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # For dim=-2, CB sizes depend on Ht and chunk_size
    cb_input_id = 0
    cb_scaler_id = 2
    cb_out_id = 16
    cb_max_id = 24
    cb_exp_id = 25
    cb_recip_id = 26

    # The max chunk_size any core will process
    max_chunk = min(
        chunk_size, max(units_per_core_g1, units_per_core_g2) if units_per_core_g2 > 0 else units_per_core_g1
    )

    cbs = []

    # CB 0: input - Ht * max_chunk pages
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Ht * max_chunk * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_input_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 2: reduce scaler - 1 page (persistent)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_scaler_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 16: output - 2 pages (double-buffered)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_out_id,
                    data_format=output_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 24: max per column - max_chunk pages
    cbs.append(
        ttnn.CBDescriptor(
            total_size=max_chunk * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_max_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 25: exp values - Ht * max_chunk pages
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Ht * max_chunk * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_exp_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # CB 26: reciprocal of sum per column - max_chunk pages
    cbs.append(
        ttnn.CBDescriptor(
            total_size=max_chunk * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_recip_id,
                    data_format=input_tensor.dtype,
                    page_size=page_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel (dim=-2) ---
    scaler_bits = _pack_two_bfloat16(1.0)
    reader_ct_args = [scaler_bits]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = _build_runtime_args_reader_h(
        input_tensor,
        compute_grid,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
        Ht,
        Wt,
        NC,
        chunk_size,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_reader_h.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel (dim=-2) ---
    # Compile-time args: Ht, chunk_size (Wt for compute), NC=1, numeric_stable
    compute_ct_args = [Ht, chunk_size, 1, 1 if numeric_stable else 0]

    compute_rt_args = _build_runtime_args_compute(
        compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_compute_h.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [cb_out_id]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = _build_runtime_args_writer_h(
        output_tensor,
        compute_grid,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
        Ht,
        Wt,
        NC,
        chunk_size,
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_writer_h.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )


# ========== RUNTIME ARGS BUILDERS ==========


def _build_runtime_args_reader_w(
    input_tensor, compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2, Wt
):
    """Build per-core runtime args for dim=-1 reader.

    Runtime args per core:
        [0] src_addr      - Input buffer DRAM address
        [1] num_tiles     - Total input tiles for this core (units * Wt)
        [2] start_id      - First tile index in flat tile array
    """
    rt_args = ttnn.RuntimeArgs()
    src_addr = input_tensor.buffer_address()
    current_unit = 0

    # Core group 1
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_tiles = units_per_core_g1 * Wt
                start_tile = current_unit * Wt
                rt_args[x][y] = [src_addr, num_tiles, start_tile]
                current_unit += units_per_core_g1

    # Core group 2
    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_tiles = units_per_core_g2 * Wt
                start_tile = current_unit * Wt
                rt_args[x][y] = [src_addr, num_tiles, start_tile]
                current_unit += units_per_core_g2

    # Set empty args for idle cores in the grid
    _fill_idle_cores(rt_args, compute_grid, core_group_1, core_group_2)

    return rt_args


def _build_runtime_args_writer_w(
    output_tensor, compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2, Wt
):
    """Build per-core runtime args for dim=-1 writer.

    Runtime args per core:
        [0] dst_addr      - Output buffer DRAM address
        [1] num_pages     - Total output tiles for this core
        [2] start_id      - First output tile index
    """
    rt_args = ttnn.RuntimeArgs()
    dst_addr = output_tensor.buffer_address()
    current_unit = 0

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_pages = units_per_core_g1 * Wt
                start_id = current_unit * Wt
                rt_args[x][y] = [dst_addr, num_pages, start_id]
                current_unit += units_per_core_g1

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_pages = units_per_core_g2 * Wt
                start_id = current_unit * Wt
                rt_args[x][y] = [dst_addr, num_pages, start_id]
                current_unit += units_per_core_g2

    _fill_idle_cores(rt_args, compute_grid, core_group_1, core_group_2)
    return rt_args


def _build_runtime_args_reader_h(
    input_tensor, compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2, Ht, Wt, NC, chunk_size
):
    """Build per-core runtime args for dim=-2 reader.

    Runtime args per core:
        [0] src_addr          - Input buffer DRAM address
        [1] num_tiles         - Total input tiles for this core (num_cols * Ht)
        [2] start_id          - First tile index (row-major flat)
        [3] col_start_tile_id - Starting tile in flat array for the first column
        [4] curr_col_in_batch - Starting column index within the current batch
        [5] num_cols          - Number of columns assigned to this core
        [6] Wt                - Full tensor width in tiles (row stride)
        [7] Ht                - Tile rows per batch element
        [8] chunk_size        - Max columns per chunk
    """
    rt_args = ttnn.RuntimeArgs()
    src_addr = input_tensor.buffer_address()
    current_col = 0  # global column counter across all batches

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_cols = units_per_core_g1
                num_tiles = num_cols * Ht
                # Compute batch and column within batch
                batch_idx = current_col // Wt
                col_in_batch = current_col % Wt
                # col_start_tile_id: starting tile in flat layout
                col_start_tile_id = batch_idx * (Ht * Wt) + col_in_batch
                start_id = col_start_tile_id
                rt_args[x][y] = [
                    src_addr,
                    num_tiles,
                    start_id,
                    col_start_tile_id,
                    col_in_batch,
                    num_cols,
                    Wt,
                    Ht,
                    chunk_size,
                ]
                current_col += num_cols

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_cols = units_per_core_g2
                num_tiles = num_cols * Ht
                batch_idx = current_col // Wt
                col_in_batch = current_col % Wt
                col_start_tile_id = batch_idx * (Ht * Wt) + col_in_batch
                start_id = col_start_tile_id
                rt_args[x][y] = [
                    src_addr,
                    num_tiles,
                    start_id,
                    col_start_tile_id,
                    col_in_batch,
                    num_cols,
                    Wt,
                    Ht,
                    chunk_size,
                ]
                current_col += num_cols

    _fill_idle_cores(rt_args, compute_grid, core_group_1, core_group_2)
    return rt_args


def _build_runtime_args_writer_h(
    output_tensor,
    compute_grid,
    core_group_1,
    core_group_2,
    units_per_core_g1,
    units_per_core_g2,
    Ht,
    Wt,
    NC,
    chunk_size,
):
    """Build per-core runtime args for dim=-2 writer.

    Writer maps tiles from chunked column order back to correct DRAM positions.
    Runtime args per core:
        [0] dst_addr          - Output buffer DRAM address
        [1] num_pages         - Total output tiles for this core
        [2] start_id          - First output tile index (batch_base + col_in_batch)
        [3] curr_col_in_batch - Starting column index within the current batch
        [4] num_cols          - Number of columns assigned to this core
        [5] Wt                - Full tensor width in tiles (row stride)
        [6] Ht                - Tile rows per batch element
        [7] chunk_size        - Max columns per chunk
    """
    rt_args = ttnn.RuntimeArgs()
    dst_addr = output_tensor.buffer_address()
    current_col = 0

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_cols = units_per_core_g1
                num_pages = num_cols * Ht
                batch_idx = current_col // Wt
                col_in_batch = current_col % Wt
                start_id = batch_idx * (Ht * Wt) + col_in_batch
                rt_args[x][y] = [dst_addr, num_pages, start_id, col_in_batch, num_cols, Wt, Ht, chunk_size]
                current_col += num_cols

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                num_cols = units_per_core_g2
                num_pages = num_cols * Ht
                batch_idx = current_col // Wt
                col_in_batch = current_col % Wt
                start_id = batch_idx * (Ht * Wt) + col_in_batch
                rt_args[x][y] = [dst_addr, num_pages, start_id, col_in_batch, num_cols, Wt, Ht, chunk_size]
                current_col += num_cols

    _fill_idle_cores(rt_args, compute_grid, core_group_1, core_group_2)
    return rt_args


def _build_runtime_args_compute(compute_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2):
    """Build per-core runtime args for compute kernel.

    Compute kernel receives number of work units (rows for dim=-1, chunks for dim=-2).
    Runtime args per core:
        [0] num_units - Number of work units for this core
    """
    rt_args = ttnn.RuntimeArgs()

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                rt_args[x][y] = [units_per_core_g1]

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                rt_args[x][y] = [units_per_core_g2]

    _fill_idle_cores(rt_args, compute_grid, core_group_1, core_group_2)
    return rt_args


def _fill_idle_cores(rt_args, compute_grid, core_group_1, core_group_2):
    """Set empty runtime args for all idle cores in the compute grid.

    CRITICAL: All cores in the grid must have runtime args set, even idle ones.
    """
    active_cores = set()
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                active_cores.add((x, y))
    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                active_cores.add((x, y))

    for x in range(compute_grid.x):
        for y in range(compute_grid.y):
            if (x, y) not in active_cores:
                rt_args[x][y] = []
