# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for the softmax operation.

Builds the CB layout, kernel descriptors, and runtime args described in
``op_design.md``:

    Numeric-stable=True (4 phases):
        Phase A: reduce<MAX, ...,  WaitUpfrontNoPop>(cb_input_tiles, cb_max_scaler, cb_max)
        Phase B: sub<bcast, ..., WaitUpfrontPopAtEnd>(cb_input_tiles, cb_max, cb_exps) + exp postop
        Phase C: reduce<SUM, ..., WaitUpfrontNoPop>(cb_exps, cb_sum_scaler, cb_inv_sum) + recip postop
        Phase D: mul<bcast, ..., WaitUpfrontPopAtEnd>(cb_exps, cb_inv_sum, cb_output_tiles)

    Numeric-stable=False (2 phases):
        Phase B′: sfpu_exp<cb_input_tiles>(cb_exps, reduce_dim_tiles)
        Phase C, D: as above.

Work distribution: one work-item = one "reduce strip" (1 × Wt for dim=-1,
Ht × 1 for dim=-2). Strips are split across the full compute_with_storage
grid via `ttnn.split_work_to_cores`. Per-core counts are passed via runtime
args (so a single kernel binary handles both core groups).
"""

from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


# CB index ranges:
#   0–7   input
#   8–15  special (scalers, constants)
#   16–23 output
#   24–31 intermediate
#
# All names below are semantic, per repo policy.
CB_INPUT_TILES = 0
CB_MAX_SCALER = 8
CB_SUM_SCALER = 9
CB_OUTPUT_TILES = 16
CB_MAX = 24
CB_EXPS = 25
CB_INV_SUM = 26


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    dim: int,
    numeric_stable: bool,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    # ------- Tensor metadata -------
    shape = list(input_tensor.shape)
    n, c, h, w = shape

    nc = n * c
    Ht = h // TILE_DIM
    Wt = w // TILE_DIM

    # Reduce-strip definition (see op_design.md, "Reduce-strip definition").
    # dim=-1 → strip = 1×Wt tiles, num_strips = NC*Ht, reduce_dim_tiles = Wt.
    # dim=-2 → strip = Ht×1 tiles, num_strips = NC*Wt, reduce_dim_tiles = Ht.
    if dim == -1:
        num_strips = nc * Ht
        reduce_dim_tiles = Wt
    elif dim == -2:
        num_strips = nc * Wt
        reduce_dim_tiles = Ht
    else:
        raise ValueError(f"softmax program descriptor: unsupported dim={dim}")

    input_page_size = input_tensor.buffer_page_size()  # fp32 tile = 4096 B
    output_page_size = output_tensor.buffer_page_size()  # fp32 tile = 4096 B
    scaler_page_size = ttnn.tile_size(ttnn.bfloat16)  # bf16 tile = 2048 B

    # ------- Core grid + work distribution -------
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    (
        num_cores_total,
        all_cores,
        core_group_1,
        core_group_2,
        strips_per_core_group_1,
        strips_per_core_group_2,
    ) = ttnn.split_work_to_cores(grid, num_strips)

    # ------- Circular buffer descriptors -------
    # See op_design.md "Circular Buffers" table for sizing rationale.
    #
    #   cb_input_tiles : 2 × reduce_dim_tiles  → double-block, reader/compute pipelining.
    #   cb_max_scaler  : 1                     → persistent, NoPop.
    #   cb_sum_scaler  : 1                     → persistent, NoPop.
    #   cb_output_tiles: 2 × reduce_dim_tiles  → double-block, compute/writer pipelining.
    #   cb_max         : 2                     → single tile per strip, 2 pages for cross-strip overlap.
    #   cb_exps        : reduce_dim_tiles      → must hold a full strip (Phase B→C sequential helpers).
    #   cb_inv_sum     : 2                     → single tile per strip, 2 pages for cross-strip overlap.

    cb_input_tiles_pages = 2 * reduce_dim_tiles
    cb_output_tiles_pages = 2 * reduce_dim_tiles
    cb_max_pages = 2
    cb_exps_pages = reduce_dim_tiles
    cb_inv_sum_pages = 2

    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_input_tiles_pages * input_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=1 * scaler_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MAX_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=1 * scaler_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SUM_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_output_tiles_pages * output_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=output_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_max_pages * input_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MAX,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_exps_pages * input_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EXPS,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_inv_sum_pages * input_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INV_SUM,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        ),
    ]

    # ------- Compile-time args (shared across all cores) -------
    # dim_is_row: 1 for dim=-1 (REDUCE_ROW + BroadcastDim::COL), 0 for dim=-2 (REDUCE_COL + ROW).
    dim_is_row = 1 if dim == -1 else 0
    numeric_stable_flag = 1 if numeric_stable else 0

    # Reader CT args:
    #   [0] dim_is_row             — selects pool-type/reduce-dim-aware scaler overload
    #   [1] Ht
    #   [2] Wt
    #   [3] reduce_dim_tiles
    #   [4..] TensorAccessorArgs(input_tensor)
    reader_ct_args = [
        dim_is_row,
        Ht,
        Wt,
        reduce_dim_tiles,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Writer CT args:
    #   [0] dim_is_row
    #   [1] Ht
    #   [2] Wt
    #   [3] reduce_dim_tiles
    #   [4..] TensorAccessorArgs(output_tensor)
    writer_ct_args = [
        dim_is_row,
        Ht,
        Wt,
        reduce_dim_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute CT args:
    #   [0] dim_is_row
    #   [1] numeric_stable
    #   [2] Ht
    #   [3] Wt
    #   [4] reduce_dim_tiles
    compute_ct_args = [
        dim_is_row,
        numeric_stable_flag,
        Ht,
        Wt,
        reduce_dim_tiles,
    ]

    # ------- Runtime args (per-core: strip count + starting strip index) -------
    # Reader / Writer:
    #   [0] base buffer address
    #   [1] num_strips_for_this_core
    #   [2] start_strip_id
    #
    # Compute:
    #   [0] num_strips_for_this_core

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()

    strip_cursor = 0
    for group_cores, strips_per_core in (
        (core_group_1, strips_per_core_group_1),
        (core_group_2, strips_per_core_group_2),
    ):
        if strips_per_core == 0:
            continue
        for core_range in group_cores.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [input_addr, strips_per_core, strip_cursor]
                    writer_rt_args[x][y] = [output_addr, strips_per_core, strip_cursor]
                    compute_rt_args[x][y] = [strips_per_core]
                    strip_cursor += strips_per_core

    assert (
        strip_cursor == num_strips
    ), f"softmax: dispatched {strip_cursor} strips, expected {num_strips} (split_work_to_cores mismatch)"

    # ------- Kernels -------
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
