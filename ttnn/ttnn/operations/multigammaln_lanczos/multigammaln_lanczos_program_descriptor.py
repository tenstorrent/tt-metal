# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
multigammaln_lanczos — ProgramDescriptor.

CB layout (see op_design.md):
    cb_input_tiles      (0)   — reader → compute, double-buffered (2 pages)
    cb_output_tiles     (16)  — compute → writer, double-buffered (2 pages)
    cb_accumulator      (24)  — compute → compute, intra-tile RMW (2 pages)

Work distribution:
    - Per-tile elementwise: each input tile is one independent work unit.
    - ``ttnn.split_work_to_cores(grid_size, total_tiles)`` partitions work
      across the compute_with_storage grid. Group 1 cores get
      ``tiles_per_core_g1`` tiles, group 2 (if non-empty) get
      ``tiles_per_core_g2`` tiles.
    - Each kernel has a single descriptor scoped to ``all_cores``; per-core
      ``num_tiles`` and ``start_tile_id`` are passed as runtime args.

Compute config (hard-coded per Phase 0 spec):
    math_fidelity   = HiFi4
    fp32_dest_acc_en = True
"""

from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    # ========== 1. Tensor metadata ==========
    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    total_tiles = input_tensor.buffer_num_pages()

    # ========== 2. Work distribution ==========
    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tiles_per_core_g1,
        tiles_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_tiles)

    # ========== 3. Circular Buffers ==========
    # CB indices follow convention: 0-7 input, 16-23 output, 24-31 intermediate.
    CB_INPUT_TILES = 0
    CB_OUTPUT_TILES = 16
    CB_ACCUMULATOR = 24

    # Intermediate accumulator is float32 (matches input/output) — sized for
    # the read-modify-write cycle that holds the front (previous accumulator)
    # and the back (new accumulator) simultaneously. Two pages is the minimum.
    accumulator_page_size = ttnn.tile_size(ttnn.float32)

    cb_input_tiles_descriptor = ttnn.CBDescriptor(
        total_size=2 * input_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILES,
                data_format=input_tensor.dtype,
                page_size=input_page_size,
            )
        ],
    )

    cb_output_tiles_descriptor = ttnn.CBDescriptor(
        total_size=2 * output_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=output_page_size,
            )
        ],
    )

    cb_accumulator_descriptor = ttnn.CBDescriptor(
        total_size=2 * accumulator_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_ACCUMULATOR,
                data_format=ttnn.float32,
                page_size=accumulator_page_size,
            )
        ],
    )

    # ========== 4. Per-core runtime arg assignment ==========
    # Walk the cores in the same order split_work_to_cores walks them: group_1
    # first (cores with more work), then group_2.

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    current_tile = 0
    for group, tiles_per_core in (
        (core_group_1, tiles_per_core_g1),
        (core_group_2, tiles_per_core_g2),
    ):
        if tiles_per_core == 0:
            continue
        for core_range in group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        tiles_per_core,
                        current_tile,
                    ]
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        tiles_per_core,
                        current_tile,
                    ]
                    compute_rt_args[x][y] = [tiles_per_core]
                    current_tile += tiles_per_core

    # ========== 5. Kernels ==========
    # Reader: scalar CT args first, then TensorAccessorArgs.
    reader_ct_args = [CB_INPUT_TILES]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "multigammaln_lanczos_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct_args = [CB_OUTPUT_TILES]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "multigammaln_lanczos_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct_args = [CB_INPUT_TILES, CB_OUTPUT_TILES, CB_ACCUMULATOR]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "multigammaln_lanczos_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[
            cb_input_tiles_descriptor,
            cb_output_tiles_descriptor,
            cb_accumulator_descriptor,
        ],
    )
