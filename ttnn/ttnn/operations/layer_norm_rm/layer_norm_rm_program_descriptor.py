# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for layer_norm_rm.

Per op_design.md (Phase 0):
- Work unit = one tile-row (32 contiguous sticks of width W → Wt tiles after
  tilization).
- Total work items = NC * Ht where NC = prod(shape[:-2]), Ht = H/32.
- Grid = full Wormhole compute_with_storage_grid; split_work_to_cores
  apportions the tile-rows. Cores are otherwise independent (no semaphores,
  no multicast).
- CB layout mirrors the design table (see op_design.md §"Circular Buffers").

Advisory deviations vs the design table:
- `cb_scaler` is fp32, not bf16 — softmax found the bf16 path silently lost
  precision on fp32 inputs (~3e-3 relative). prepare_reduce_scaler deduces
  format from the CB so making the CB fp32 preserves the full SrcA precision
  through the multiply-accumulate. Doubles cb_scaler from 2 KB to 4 KB —
  negligible vs the per-core L1 budget.
- compute_kernel_hw_startup uses the 3-arg form (input_rm, scaler, output)
  to be conservative across the chained helpers.
"""

import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


# Semantic CB indices (per op_design.md):
#   0-7   input
#   8-15  special (scalers, constants)
#   16-23 output
#   24-31 intermediate
CB_INPUT_RM = 0
CB_GAMMA_RM = 1
CB_BETA_RM = 2
CB_SCALER = 8
CB_GAMMA_TILES = 9
CB_BETA_TILES = 10
CB_OUTPUT_TILES = 16
CB_INPUT_TILES = 24
CB_MEAN = 25
CB_CENTERED = 26
CB_CENTERED_SQ = 27
CB_INV_STD = 28
CB_NORM = 29


def _fp32_bits(value: float) -> int:
    """Bit-cast an fp32 value to its uint32 representation (little-endian)."""
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    # ------------------------------------------------------------------
    # Tensor metadata
    # ------------------------------------------------------------------
    shape = list(input_tensor.shape)
    H = shape[-2]
    W = shape[-1]
    NC = 1
    for d in shape[:-2]:
        NC *= d

    Ht = H // TILE_DIM
    Wt = W // TILE_DIM
    num_tile_rows = NC * Ht

    elem_size = input_tensor.element_size()
    row_bytes = W * elem_size  # bytes per logical row (= per stick)
    tile_size_f32 = ttnn.tile_size(ttnn.float32)  # 4096 B

    has_gamma = gamma is not None
    has_beta = beta is not None

    # ------------------------------------------------------------------
    # Compute config — Phase 0 hard-codes HiFi4 + fp32 dest accumulator
    # (per design "Compute config policy").
    # ------------------------------------------------------------------
    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )

    # ------------------------------------------------------------------
    # Core grid + work distribution
    # ------------------------------------------------------------------
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    (
        num_cores_total,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, num_tile_rows)

    # ------------------------------------------------------------------
    # Circular buffer descriptors
    # ------------------------------------------------------------------
    cbs = []

    # cb_input_rm — reader fills via read_sticks_for_tilize<TILE>.
    # 2 * Wt pages double-buffers a full tile-row.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * Wt * tile_size_f32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_RM,
                    data_format=ttnn.float32,
                    page_size=tile_size_f32,
                )
            ],
        )
    )

    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size_f32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_RM,
                        data_format=ttnn.float32,
                        page_size=tile_size_f32,
                    )
                ],
            )
        )
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size_f32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_RM,
                        data_format=ttnn.float32,
                        page_size=tile_size_f32,
                    )
                ],
            )
        )

    # cb_scaler — fp32 (advisory deviation from design's bf16 — see file docstring).
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size_f32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.float32,
                    page_size=tile_size_f32,
                )
            ],
        )
    )

    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size_f32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_TILES,
                        data_format=ttnn.float32,
                        page_size=tile_size_f32,
                    )
                ],
            )
        )
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size_f32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_TILES,
                        data_format=ttnn.float32,
                        page_size=tile_size_f32,
                    )
                ],
            )
        )

    # cb_output_tiles — fp32, double-buffered.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * Wt * tile_size_f32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=ttnn.float32,
                    page_size=tile_size_f32,
                )
            ],
        )
    )

    # Intermediate CBs — sized to hold a full tile-row each (Wt) since the
    # downstream helpers are sequential and the producer fills the strip
    # before the consumer starts.
    for cb_idx, num_pages in (
        (CB_INPUT_TILES, Wt),
        (CB_MEAN, 2),
        (CB_CENTERED, Wt),
        (CB_CENTERED_SQ, Wt),
        (CB_INV_STD, 2),
        (CB_NORM, Wt),
    ):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * tile_size_f32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_idx,
                        data_format=ttnn.float32,
                        page_size=tile_size_f32,
                    )
                ],
            )
        )

    # ------------------------------------------------------------------
    # Reader CT args + RT args
    # ------------------------------------------------------------------
    # Scalar CT args:
    #   [0] row_bytes
    #   [1] W
    #   [2] Wt
    #   [3] has_gamma
    #   [4] has_beta
    # Then chained TensorAccessorArgs: input, gamma, beta (gamma/beta use
    # no-arg placeholders when absent).
    reader_ct_args = [
        row_bytes,
        W,
        Wt,
        1 if has_gamma else 0,
        1 if has_beta else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    # ------------------------------------------------------------------
    # Writer CT args (just row_bytes + output accessor)
    # ------------------------------------------------------------------
    writer_ct_args = [row_bytes]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ------------------------------------------------------------------
    # Compute CT args
    # ------------------------------------------------------------------
    compute_ct_args = [
        Wt,
        1 if has_gamma else 0,
        1 if has_beta else 0,
        _fp32_bits(epsilon),  # passed as uint32 bit pattern, fed to add_unary_tile
    ]

    # ------------------------------------------------------------------
    # Per-core runtime args
    # ------------------------------------------------------------------
    #   reader: [input_addr, gamma_addr, beta_addr, start_tile_row, num_tile_rows]
    #   writer: [output_addr, start_tile_row, num_tile_rows]
    #   compute: [num_tile_rows]
    #
    # gamma_addr / beta_addr are 0 when absent; the kernel guards on has_gamma /
    # has_beta CT flags before touching the corresponding accessor.
    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    cursor = 0
    for group_cores, rows_per_core in (
        (core_group_1, rows_per_core_g1),
        (core_group_2, rows_per_core_g2),
    ):
        if rows_per_core == 0:
            continue
        for core_range in group_cores.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_addr,
                        gamma_addr,
                        beta_addr,
                        cursor,
                        rows_per_core,
                    ]
                    writer_rt_args[x][y] = [output_addr, cursor, rows_per_core]
                    compute_rt_args[x][y] = [rows_per_core]
                    cursor += rows_per_core

    assert cursor == num_tile_rows, (
        f"layer_norm_rm: dispatched {cursor} tile-rows, expected {num_tile_rows} " f"(split_work_to_cores mismatch)"
    )

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
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
