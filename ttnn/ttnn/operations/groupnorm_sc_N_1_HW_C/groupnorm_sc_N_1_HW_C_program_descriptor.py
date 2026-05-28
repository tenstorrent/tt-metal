# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ProgramDescriptor for groupnorm_sc_N_1_HW_C.

Single-core kernel implementing GroupNorm on (N, 1, HW, C) tensors. Handles
both TILE_LAYOUT and ROW_MAJOR_LAYOUT inputs and the (C / num_groups) % 32 != 0
case via kernel-internal masking.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices — semantic names tied to numeric slots
CB_INPUT_RM_R = 0  # RM input sticks (phase R), RM input only
CB_INPUT_RM_A = 1  # RM input sticks (phase A), RM input only
CB_INPUT_TILES_R = 2  # tiled input phase R
CB_INPUT_TILES_A = 3  # tiled input phase A
CB_GAMMA_RM = 4  # gamma sticks, replicated 32×
CB_BETA_RM = 5  # beta sticks, replicated 32×
CB_GAMMA_TILE = 6  # tilized gamma (one tile per output T)
CB_BETA_TILE = 7  # tilized beta (one tile per output T)
CB_SCALER_ONE = 8  # reduce-scaler tile (1.0 for SUM REDUCE_SCALAR)
CB_MASK_STREAM = 9  # streaming row-replicated mask tile
CB_INV_N_SCALAR = 10  # scalar 1/N_per_g tile (scalar at (0,0))
CB_EPS_SCALAR = 11  # scalar eps tile (scalar at (0,0))
CB_RUNNING_ACC_SUM = 12  # 1-slot running accumulator for sum (phase R)
CB_RUNNING_ACC_SUMSQ = 13  # 1-slot running accumulator for sumsq (phase R)
CB_OUTPUT_TILES = 16  # output to writer
CB_ACTIVE_MEAN = 24  # 1-tile active mean for the current g (phase A)
CB_ACTIVE_RCP_STD = 25  # 1-tile active rcp_std for the current g (phase A)
CB_GROUP_MEAN = 26  # per-group mean (G tiles)
CB_GROUP_RCP_STD = 27  # per-group rcp_std (G tiles)
CB_SCRATCH_A = 28  # scratch a
CB_SCRATCH_B = 29  # scratch b
CB_MEANS_TILE_T = 30  # row-replicated means tile for current output T
CB_RCP_STD_TILE_T = 31  # row-replicated rcp_std tile for current output T


def _float_bits(f: float) -> int:
    """Return the IEEE-754 single-precision bit pattern as a uint32."""
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def _input_layout_code(layout) -> int:
    if layout == ttnn.TILE_LAYOUT:
        return 0
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return 1
    raise ValueError(f"Unsupported input layout: {layout}")


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_groups: int,
    gamma: Optional[ttnn.Tensor],
    beta: Optional[ttnn.Tensor],
    eps: float,
) -> ttnn.ProgramDescriptor:
    # ============================================================
    # Geometry
    # ============================================================
    shape = list(input_tensor.shape)
    N, _one, HW, C = shape
    G = int(num_groups)
    Cg = C // G
    Ht = HW // 32  # design enforces HW % 32 == 0 in Phase 0
    Ct = (C + 31) // 32

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0
    input_layout_code = _input_layout_code(input_tensor.layout)

    eps_bits = _float_bits(eps)

    # Stick size for the activation tensor (RM page size).
    bf16_elem_size = 2
    stick_bytes = C * bf16_elem_size

    # Tile size for bf16
    tile_bytes = ttnn.tile_size(input_tensor.dtype)  # 2048 for bf16

    # ============================================================
    # Core grid — single core
    # ============================================================
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ============================================================
    # CB descriptors
    # ============================================================
    cbs = []

    # cb_input_rm_R / cb_input_rm_A — only configured for RM input
    if input_layout_code == 1:
        # Reader reads 32 sticks per (g, T_in_g, r) iteration; each stick
        # is one "chunk" of 32 bf16 channels = 64 bytes. Double-buffer:
        # 64 pages × 64 bytes = 4 KB per RM CB.
        rm_chunk_bytes = 32 * bf16_elem_size  # 64 bytes
        # round up to dram alignment for safety
        rm_pad_align = ttnn.get_dram_alignment()
        rm_page = ((rm_chunk_bytes + rm_pad_align - 1) // rm_pad_align) * rm_pad_align
        cbs.append(
            ttnn.CBDescriptor(
                total_size=64 * rm_page,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_RM_R,
                        data_format=input_tensor.dtype,
                        page_size=rm_page,
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=64 * rm_page,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_RM_A,
                        data_format=input_tensor.dtype,
                        page_size=rm_page,
                    )
                ],
            )
        )

    # cb_input_tiles_R / cb_input_tiles_A — 2 tiles each (double-buffered)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES_R,
                    data_format=input_tensor.dtype,
                    page_size=tile_bytes,
                )
            ],
        )
    )
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES_A,
                    data_format=input_tensor.dtype,
                    page_size=tile_bytes,
                )
            ],
        )
    )

    # gamma / beta CBs — only when present
    rm_pad_align = ttnn.get_dram_alignment()
    gamma_rm_chunk_bytes = 32 * bf16_elem_size  # 64 bytes per chunk along W
    gamma_rm_page = ((gamma_rm_chunk_bytes + rm_pad_align - 1) // rm_pad_align) * rm_pad_align
    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=32 * gamma_rm_page,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_RM,
                        data_format=gamma.dtype,
                        page_size=gamma_rm_page,
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_TILE,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=32 * gamma_rm_page,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_RM,
                        data_format=beta.dtype,
                        page_size=gamma_rm_page,
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_TILE,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )

    # Scaler / one-shot tiles
    for cb_index in (CB_SCALER_ONE, CB_INV_N_SCALAR, CB_EPS_SCALAR):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )

    # Streaming mask CB (2 pages, double-buffered)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MASK_STREAM,
                    data_format=input_tensor.dtype,
                    page_size=tile_bytes,
                )
            ],
        )
    )

    # Running accumulators (1 slot each is enough — they're popped + repushed each iter)
    for cb_index in (CB_RUNNING_ACC_SUM, CB_RUNNING_ACC_SUMSQ):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )

    # Output CB
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=tile_bytes,
                )
            ],
        )
    )

    # Per-group stat CBs — G tiles each
    for cb_index in (CB_GROUP_MEAN, CB_GROUP_RCP_STD):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=G * tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )

    # Active mean/rcp_std CBs — 2-tile each, used during phase A expansion.
    # We copy_tile cb_group_mean[g] / cb_group_rcp_std[g] here so binary_op<SCALAR>
    # can address tile 0 as required.
    for cb_index in (CB_ACTIVE_MEAN, CB_ACTIVE_RCP_STD):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )

    # Scratch CBs and per-output-tile expansion CBs
    for cb_index in (CB_SCRATCH_A, CB_SCRATCH_B):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )
    for cb_index in (CB_MEANS_TILE_T, CB_RCP_STD_TILE_T):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=input_tensor.dtype,
                        page_size=tile_bytes,
                    )
                ],
            )
        )

    # ============================================================
    # Kernel descriptors
    # ============================================================

    # Reader CT args layout (all scalars first, then TensorAccessorArgs at the end)
    #  0: input_layout_code
    #  1: has_gamma
    #  2: has_beta
    #  3: N
    #  4: HW
    #  5: C
    #  6: G
    #  7: Cg
    #  8: Ht
    #  9: Ct
    # 10: eps_bits
    # 11: stick_bytes
    # 12..: TensorAccessorArgs (input, gamma_or_placeholder, beta_or_placeholder)
    reader_ct_args = [
        input_layout_code,
        has_gamma,
        has_beta,
        N,
        HW,
        C,
        G,
        Cg,
        Ht,
        Ct,
        eps_bits,
        stick_bytes,
    ]

    input_accessor = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_ct_args.extend(input_accessor)

    if has_gamma:
        gamma_accessor = ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
    else:
        gamma_accessor = ttnn.TensorAccessorArgs().get_compile_time_args()
    reader_ct_args.extend(gamma_accessor)

    if has_beta:
        beta_accessor = ttnn.TensorAccessorArgs(beta).get_compile_time_args()
    else:
        beta_accessor = ttnn.TensorAccessorArgs().get_compile_time_args()
    reader_ct_args.extend(beta_accessor)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address() if gamma is not None else 0,
        beta.buffer_address() if beta is not None else 0,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer CT args:
    #  0: N
    #  1: Ht
    #  2: Ct
    #  3..: TensorAccessorArgs (output)
    writer_ct_args = [N, Ht, Ct]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address()]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute CT args:
    #  0: input_layout_code
    #  1: has_gamma
    #  2: has_beta
    #  3: N
    #  4: HW
    #  5: C
    #  6: G
    #  7: Cg
    #  8: Ht
    #  9: Ct
    # 10: eps_bits (fp32 bit-packed for SFPU AddScalar)
    compute_ct_args = [
        input_layout_code,
        has_gamma,
        has_beta,
        N,
        HW,
        C,
        G,
        Cg,
        Ht,
        Ct,
        eps_bits,
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
