# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
atan_mean — ProgramDescriptor.

CB layout (see op_design.md):
    cb_input_tiles      ( 0)  — reader → compute, double-buffered (2 pages, fp32)
    cb_scaler           ( 8)  — reader → compute (persistent, never popped) (1 page, bf16)
    cb_output_tiles     (16)  — compute → writer, double-buffered (2 pages, fp32)
    cb_atan_tiles       (24)  — compute → compute (intermediate) (Wt pages, fp32)

Work unit:
    One *row-tile* = one ``(n, c, ht)`` triple. Each row-tile collapses ``Wt``
    input tiles into a single output tile via SFPU atan + REDUCE_ROW AVG.
    Total = N * C * Ht. Distributed across the full compute_with_storage grid
    by ``ttnn.split_work_to_cores``.

Compute config (hard-coded per Phase 0 spec):
    math_fidelity     = HiFi4
    fp32_dest_acc_en  = True
"""

from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    # ========== 1. Tensor metadata ==========
    input_shape = list(input_tensor.shape)
    N, C, H, W = input_shape  # validated rank-4 in the entry point
    Ht = H // TILE_DIM
    Wt = W // TILE_DIM
    total_row_tiles = N * C * Ht

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    scaler_page_size = ttnn.tile_size(ttnn.bfloat16)

    # ========== 2. Work distribution ==========
    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        row_tiles_per_core_g1,
        row_tiles_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_row_tiles)

    # ========== 3. Circular Buffers ==========
    CB_INPUT_TILES = 0
    CB_SCALER = 8
    CB_OUTPUT_TILES = 16
    CB_ATAN_TILES = 24

    cb_input_tiles_descriptor = ttnn.CBDescriptor(
        total_size=2 * input_page_size,  # double-buffer streaming
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES,
                data_format=input_tensor.dtype,
                page_size=input_page_size,
            )
        ],
    )

    # cb_scaler is bf16 (matmul-path AVG/REDUCE_ROW scaler in col-0 fill).
    # 1 page — pushed once at program startup, ``cb_wait_front``-ed by every
    # ``reduce<>`` call but never popped.
    cb_scaler_descriptor = ttnn.CBDescriptor(
        total_size=scaler_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=ttnn.bfloat16,
                page_size=scaler_page_size,
            )
        ],
    )

    cb_output_tiles_descriptor = ttnn.CBDescriptor(
        total_size=2 * output_page_size,  # double-buffer compute → writer
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=output_page_size,
            )
        ],
    )

    # cb_atan_tiles: holds the full row of post-atan tiles for one row-tile.
    # ``sfpu_atan`` (sequential) pushes Wt tiles before ``reduce<>``
    # (also sequential) starts consuming, so the CB must hold Wt pages —
    # smaller would deadlock.
    cb_atan_tiles_descriptor = ttnn.CBDescriptor(
        total_size=Wt * input_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ATAN_TILES,
                data_format=input_tensor.dtype,  # fp32, same as input
                page_size=input_page_size,
            )
        ],
    )

    # ========== 4. Per-core runtime arg assignment ==========
    # Walk the cores in the same order ``split_work_to_cores`` walks them:
    # group_1 first (cores with more row-tiles), then group_2.
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    current_row_tile = 0
    for group, row_tiles_per_core in (
        (core_group_1, row_tiles_per_core_g1),
        (core_group_2, row_tiles_per_core_g2),
    ):
        if row_tiles_per_core == 0:
            continue
        for core_range in group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        row_tiles_per_core,
                        current_row_tile,
                    ]
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        row_tiles_per_core,
                        current_row_tile,
                    ]
                    compute_rt_args[x][y] = [row_tiles_per_core]
                    current_row_tile += row_tiles_per_core

    # ========== 5. Kernels ==========
    # Reader CT args: [CB_INPUT_TILES, CB_SCALER, W, TensorAccessorArgs...]
    # ``W`` is compile-time so the scaler-prep helper can templatise the
    # reduce_factor. The reader derives ``Wt = W / 32`` internally — passing
    # both would be redundant.
    reader_ct_args = [CB_INPUT_TILES, CB_SCALER, W]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "atan_mean_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer CT args: [CB_OUTPUT_TILES, TensorAccessorArgs...]
    writer_ct_args = [CB_OUTPUT_TILES]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "atan_mean_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute CT args: [CB_INPUT_TILES, CB_SCALER, CB_ATAN_TILES, CB_OUTPUT_TILES, Wt]
    compute_ct_args = [CB_INPUT_TILES, CB_SCALER, CB_ATAN_TILES, CB_OUTPUT_TILES, Wt]

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "atan_mean_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[
            cb_input_tiles_descriptor,
            cb_scaler_descriptor,
            cb_output_tiles_descriptor,
            cb_atan_tiles_descriptor,
        ],
    )
