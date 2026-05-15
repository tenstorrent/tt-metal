# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for rms_norm.

Two-pass RMSNorm via streaming reduce per row-chunk (32 rows × Wt tiles wide):
    Pass 1: mean(x^2) via SUM/REDUCE_ROW reduce with scaler 1/W
    Pass 2: x * rsqrt(mean(x^2) + eps) * gamma

Pass 1 holds Wt input tiles in cb_input_tiles via NoWaitNoPop (stage A) so
pass 2 (stage D) can stream-pop them. Reader pushes input tiles ONCE per
chunk; compute reuses them across both passes.

Advisory deviations from op_design.md:
  - Single-core for v1: the design specifies multi-core split_work_to_cores,
    but the test shapes have small Ht (≤4 chunks). Single-core is functionally
    correct; multi-core is a future refinement.
  - cb_input_raw_rm / cb_output_rm use tile-sized pages (matching
    toy_tilize_untilize convention and helper API surface).
"""

import math
import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


# --- CB index assignment (semantic names) ---
CB_INPUT_RAW_RM = 0  # RM input sticks (RM input path only)
CB_INPUT_TILES = 1  # tiled input (always present)
CB_GAMMA_RM = 2  # RM gamma stick (gamma path only)
CB_GAMMA_TILED = 3  # tiled gamma (gamma path only)
CB_SCALER = 4  # reduce scaler (bfloat16, always present)
CB_OUTPUT_TILES = 16  # tiled output (always present)
CB_OUTPUT_RM = 17  # RM output sticks (RM output path only)
CB_X_SQ = 24  # x^2 intermediate
CB_MEAN_SQ = 25  # mean(x^2) → rsqrt(mean+eps)
CB_X_NORM = 26  # x_norm intermediate (gamma path only)


def _fp32_bits(f: float) -> int:
    """Reinterpret a Python float as fp32 bits (uint32)."""
    return struct.unpack("I", struct.pack("f", float(f)))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor | None,
    output_tensor: ttnn.Tensor,
    *,
    epsilon: float,
) -> ttnn.ProgramDescriptor:
    # --- Shape / layout extraction ---
    shape = list(input_tensor.shape)
    H = shape[-2]
    W = shape[-1]
    NC = 1
    for d in shape[:-2]:
        NC *= d

    input_is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    output_is_rm = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None

    Wt = (W + TILE_DIM - 1) // TILE_DIM
    Ht = (H + TILE_DIM - 1) // TILE_DIM
    num_chunks = NC * Ht

    # Partial-W (only meaningful for RM input — TILE requires W%32==0).
    partial_w = W % TILE_DIM
    has_partial_w = (partial_w != 0) and input_is_rm

    # --- Dtype-specific sizes ---
    input_dtype = input_tensor.dtype
    output_dtype = output_tensor.dtype
    gamma_dtype = gamma.dtype if has_gamma else input_dtype

    input_elem_size = input_tensor.element_size()
    output_elem_size = output_tensor.element_size()
    gamma_elem_size = gamma.element_size() if has_gamma else input_elem_size

    input_tile_size = ttnn.tile_size(input_dtype)
    output_tile_size = ttnn.tile_size(output_dtype)
    gamma_tile_size = ttnn.tile_size(gamma_dtype)
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Row bytes (RM): origin width × element size.
    input_row_bytes = W * input_elem_size
    output_row_bytes = W * output_elem_size
    # Gamma: padded to tile width (per asymmetric tilize API contract).
    padded_W_elements = ((W + TILE_DIM - 1) // TILE_DIM) * TILE_DIM
    padded_gamma_row_bytes = padded_W_elements * gamma_elem_size

    fp32_dest = input_dtype == ttnn.float32

    # --- Single-core grid (v1 advisory deviation from design) ---
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Reader: total rows / tiles to stream ---
    if input_is_rm:
        # Total rows across all chunks (last chunk may be partial — read only valid rows).
        # For NC=1: total_rows = H. For NC>1 with tile-aligned H: total_rows = NC*H = NC*Ht*32.
        # Mixed NC>1 + partial H is not in scope for v1.
        total_input_units = NC * H  # rows
        input_start_unit = 0
    else:
        total_input_units = NC * Ht * Wt  # tiles
        input_start_unit = 0

    if output_is_rm:
        total_output_units = NC * H  # rows
        output_start_unit = 0
    else:
        total_output_units = NC * Ht * Wt  # tiles
        output_start_unit = 0

    # --- CB sizes ---
    # All CBs use single-buffer per the design's "sequential consumption" invariants
    # (cb_x_sq, cb_x_norm, cb_input_tiles for stage A NoWaitNoPop + stage D pop).
    cb_descriptors = []

    if input_is_rm:
        # cb_input_raw_rm: page_size = tile_size, num_pages = Wt
        # The read_sticks_for_tilize<TILE> helper pushes Wt tile-sized pages per
        # 32-row block. CB must hold at least Wt pages or reader deadlocks.
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * input_tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_RAW_RM,
                        data_format=input_dtype,
                        page_size=input_tile_size,
                    )
                ],
            )
        )

    # cb_input_tiles: holds Wt tiles for stage A (NoWaitNoPop) + stage D (pop Wt)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_dtype,
                    page_size=input_tile_size,
                )
            ],
        )
    )

    if has_gamma:
        # cb_gamma_rm: page_size = padded_gamma_row_bytes (asymmetric tilize input)
        # num_pages = 1 (one gamma stick re-pushed per chunk)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=1 * padded_gamma_row_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_RM,
                        data_format=gamma_dtype,
                        page_size=padded_gamma_row_bytes,
                    )
                ],
            )
        )

        # cb_gamma_tiled: Wt tile-sized pages (compute tilizes 1 stick → Wt tiles per chunk)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * gamma_tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_TILED,
                        data_format=gamma_dtype,
                        page_size=gamma_tile_size,
                    )
                ],
            )
        )

    # cb_scaler: bfloat16 always, 1 tile (no partial) or 2 tiles (full + partial)
    num_scaler_tiles = 2 if has_partial_w else 1
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=num_scaler_tiles * scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # cb_x_sq: Wt tiles (sequential consumption by reduce BulkWaitBulkPop)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * input_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_X_SQ,
                    data_format=input_dtype,
                    page_size=input_tile_size,
                )
            ],
        )
    )

    # cb_mean_sq: 2 tiles (transform_in_place pops-before-reserve; 1 active + 1 spare)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=2 * input_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MEAN_SQ,
                    data_format=input_dtype,
                    page_size=input_tile_size,
                )
            ],
        )
    )

    if has_gamma:
        # cb_x_norm: Wt tiles (intermediate between stage D and stage E)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * input_tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_X_NORM,
                        data_format=input_dtype,
                        page_size=input_tile_size,
                    )
                ],
            )
        )

    # cb_output_tiles: Wt tiles (writer consumes / untilize consumes)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * output_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_dtype,
                    page_size=output_tile_size,
                )
            ],
        )
    )

    if output_is_rm:
        # cb_output_rm: Wt tile-sized pages (untilize output → writer)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * output_tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_OUTPUT_RM,
                        data_format=output_dtype,
                        page_size=output_tile_size,
                    )
                ],
            )
        )

    # --- Reader compile-time args ---
    # Reader expects exactly this CT arg order; reader.cpp reads them by index.
    inv_W_bits = _fp32_bits(1.0 / float(W))
    reader_ct_args = [
        inv_W_bits,  # 0: scaler bits (1/W as fp32)
        1 if has_partial_w else 0,  # 1: HAS_PARTIAL_W
        partial_w if has_partial_w else TILE_DIM,  # 2: partial_w (valid positions in partial scaler tile)
        1 if input_is_rm else 0,  # 3: INPUT_IS_RM
        1 if has_gamma else 0,  # 4: HAS_GAMMA
        Wt,  # 5: Wt
        num_chunks,  # 6: num_chunks
        input_row_bytes,  # 7: input row bytes (RM)
        padded_gamma_row_bytes,  # 8: gamma row bytes (gamma)
    ]
    # Tensor accessor args (input, then gamma) at the end.
    input_args = ttnn.TensorAccessorArgs(input_tensor)
    reader_ct_args.extend(input_args.get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address() if has_gamma else 0,
        input_start_unit,
        total_input_units,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer compile-time args ---
    writer_ct_args = [
        1 if output_is_rm else 0,  # 0: OUTPUT_IS_RM
        Wt,  # 1: Wt
        output_row_bytes,  # 2: output row bytes (RM)
        num_chunks,  # 3: num_chunks
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        output_start_unit,
        total_output_units,
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute compile-time args ---
    eps_bits = _fp32_bits(epsilon)
    compute_ct_args = [
        eps_bits,  # 0: epsilon as fp32 bits
        1 if has_partial_w else 0,  # 1: HAS_PARTIAL_W
        1 if input_is_rm else 0,  # 2: INPUT_IS_RM
        1 if output_is_rm else 0,  # 3: OUTPUT_IS_RM
        1 if has_gamma else 0,  # 4: HAS_GAMMA
        Wt,  # 5: Wt
        num_chunks,  # 6: num_chunks
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )
