# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for multigammaln_lanczos (order p = 4) — Lanczos 6-term
polynomial flavour.

Architecture (single output tile is the unit of work; embarrassingly parallel):

    reader → cb_input_tiles
    compute (per tile, hold cb_input_tiles across 4 Lanczos sub-phases):
        sub-phase A x4 — Lanczos polynomial on (a - offset) for offset in
            {0.0, 0.5, 1.0, 1.5}, packed to cb_lgamma_a / _half / _one /
            _three_halves
        sub-phase B — raw FPU 4-way add + binop_with_scalar add of 3*log(pi)
            packed to cb_output_tiles
    writer ← cb_output_tiles

Compute config: HiFi4 + fp32_dest_acc_en (precision target for fp32 Lanczos).

Work distribution: split_work_to_cores over the full compute grid, one tile per
core unit. Reader, compute, and writer are placed on `all_cores`; per-core RT
args are populated by walking core_group_1 then core_group_2.
"""

from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


# CB indices (must match the kernels)
CB_INPUT_TILES = 0
CB_OUTPUT_TILES = 16
CB_LGAMMA_A = 24
CB_LGAMMA_A_HALF = 25
CB_LGAMMA_A_ONE = 26
CB_LGAMMA_A_THREE_HALVES = 27


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    # --- Tile-page geometry (fp32 tiled inputs/outputs) ---
    page_size = input_tensor.buffer_page_size()  # tile_size(float32) = 4096
    total_tiles = input_tensor.buffer_num_pages()
    out_page_size = output_tensor.buffer_page_size()

    # --- Work distribution ---
    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        pages_per_core_g1,
        pages_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_tiles)

    # --- Circular buffer descriptors ---
    # cb_input_tiles: double-buffered streaming from reader (NCRISC) to compute.
    cb_input_descriptor = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # cb_output_tiles: double-buffered streaming from compute to writer (BRISC).
    cb_output_descriptor = ttnn.CBDescriptor(
        total_size=2 * out_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=out_page_size,
            )
        ],
    )

    # Four intermediate CBs, one per per-offset Lanczos term. Each holds 1 tile
    # per input-tile iteration; double-buffer (2 pages) gives headroom for the
    # next iteration's sub-phase A to begin while sub-phase B drains the
    # previous iteration's tiles.
    intermediate_indices = [
        CB_LGAMMA_A,
        CB_LGAMMA_A_HALF,
        CB_LGAMMA_A_ONE,
        CB_LGAMMA_A_THREE_HALVES,
    ]
    intermediate_descriptors = [
        ttnn.CBDescriptor(
            total_size=2 * page_size,  # 2 pages — double buffer
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=idx,
                    data_format=input_tensor.dtype,  # fp32
                    page_size=page_size,
                )
            ],
        )
        for idx in intermediate_indices
    ]

    # --- Kernel compile-time args ---
    # Reader & writer: only TensorAccessorArgs (no scalar CT args).
    reader_ct_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute: no CT args. num_tiles is passed at runtime.
    compute_ct_args: list[int] = []

    # --- Per-core runtime args (walk core_group_1 then core_group_2) ---
    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    current_tile = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [input_addr, current_tile, pages_per_core_g1]
                writer_rt_args[x][y] = [output_addr, current_tile, pages_per_core_g1]
                compute_rt_args[x][y] = [pages_per_core_g1]
                current_tile += pages_per_core_g1
    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [input_addr, current_tile, pages_per_core_g2]
                writer_rt_args[x][y] = [output_addr, current_tile, pages_per_core_g2]
                compute_rt_args[x][y] = [pages_per_core_g2]
                current_tile += pages_per_core_g2

    # --- Kernel descriptors ---
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "multigammaln_lanczos_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "multigammaln_lanczos_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # HiFi4 + fp32 DEST accumulation — required for the precision target on
    # the multi-step Lanczos recipe (polynomial / log / recip steps).
    #
    # `unpack_to_dest_mode = UnpackToDestFp32` on every fp32 CB makes the
    # unpacker bypass the SrcA/SrcB TF32-truncation path when `copy_tile`-ing
    # an intermediate fp32 tile into DEST. Without this, the sub-phase B
    # reload of `cb_lgamma_*` can lose mantissa bits even though every CB is
    # declared Float32 and the DEST is fp32-accumulated.
    NUM_CBS = 32
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
    for cb_idx in (
        CB_INPUT_TILES,
        CB_OUTPUT_TILES,
        CB_LGAMMA_A,
        CB_LGAMMA_A_HALF,
        CB_LGAMMA_A_ONE,
        CB_LGAMMA_A_THREE_HALVES,
    ):
        unpack_modes[cb_idx] = ttnn.UnpackToDestMode.UnpackToDestFp32

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    compute_config.unpack_to_dest_mode = unpack_modes

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "multigammaln_lanczos_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_input_descriptor, cb_output_descriptor, *intermediate_descriptors],
    )
