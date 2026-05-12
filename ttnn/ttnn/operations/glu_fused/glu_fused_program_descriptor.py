# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
glu_fused — ProgramDescriptor.

CB layout (see ``op_design.md``):
    cb_input_a       (0)   — reader (first half tiles) → compute, double-buffered (2 pages)
    cb_input_b       (1)   — reader (second half tiles) → compute, double-buffered (2 pages)
    cb_output_tiles  (16)  — compute → writer, double-buffered (2 pages)

Work distribution:
    - Per output tile elementwise: each output tile is one independent work unit.
      Two distinct input tiles are read per output tile (one from each half).
    - ``ttnn.split_work_to_cores(grid_size, total_output_tiles)`` partitions output
      tiles across the compute_with_storage grid. Per-core RT args walk group_1
      first, then group_2.

Compute config is dtype-aware (Refinement 2):
    fp32 input:
        math_fidelity   = HiFi4
        fp32_dest_acc_en = True
        unpack_to_dest_mode[cb_input_a] = UnpackToDestFp32
        unpack_to_dest_mode[cb_input_b] = UnpackToDestFp32
    bf16 input:
        math_fidelity   = LoFi
        fp32_dest_acc_en = False
        (default unpack mode — UnpackToDestFp32 would zero-extend bf16,
         no precision gain, pure overhead)
"""

from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices follow convention: 0-7 input, 16-23 output.
CB_INPUT_A = 0
CB_INPUT_B = 1
CB_OUTPUT_TILES = 16


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    # ========== 1. Tensor metadata ==========
    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    total_output_tiles = output_tensor.buffer_num_pages()

    # Wt_half = (W / 32) / 2 = W / 64 — tile-cols per row of the OUTPUT, which
    # is also the number of tile-cols in each half of the INPUT. Drives the
    # split arithmetic in the reader.
    shape = list(input_tensor.shape)
    W = shape[-1]
    Wt_half = W // 64

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
    ) = ttnn.split_work_to_cores(grid_size, total_output_tiles)

    # ========== 3. Circular Buffers ==========
    cb_input_a_descriptor = ttnn.CBDescriptor(
        total_size=2 * input_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_A,
                data_format=input_tensor.dtype,
                page_size=input_page_size,
            )
        ],
    )

    cb_input_b_descriptor = ttnn.CBDescriptor(
        total_size=2 * input_page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_B,
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

    # ========== 4. Per-core runtime arg assignment ==========
    # Walk cores in the same order split_work_to_cores walks them: group_1
    # (cores with more output tiles) first, then group_2.
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
    # Reader: scalar CT args first (cb indices + Wt_half), then TensorAccessorArgs.
    reader_ct_args = [CB_INPUT_A, CB_INPUT_B, Wt_half]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "glu_fused_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer: scalar CT args first, TensorAccessorArgs at the end.
    writer_ct_args = [CB_OUTPUT_TILES]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "glu_fused_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct_args = [CB_INPUT_A, CB_INPUT_B, CB_OUTPUT_TILES]

    # Dtype-aware compute config (Refinement 2):
    #   fp32 input → HiFi4 + fp32_dest_acc + UnpackToDestFp32 on input CBs.
    #     Max-precision path. Verifier locked these in for Phase 0 fp32.
    #   bf16 input → LoFi + fp32_dest_acc=False + default unpack mode.
    #     Matches bf16's own precision regime. The fp32 settings on bf16
    #     inputs are pure overhead — UnpackToDestFp32 just zero-extends an
    #     already-bf16 value, and fp32 DEST halves the auto-batched DST
    #     parallelism. Benchmarked: fp32 settings on bf16 inputs are ~1.3×
    #     slower than the bf16-tuned settings, with no numerical benefit.
    if input_tensor.dtype == ttnn.float32:
        compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
        unpack_modes[CB_INPUT_A] = ttnn.UnpackToDestMode.UnpackToDestFp32
        unpack_modes[CB_INPUT_B] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes
    else:  # bfloat16
        compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
        )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "glu_fused_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[
            cb_input_a_descriptor,
            cb_input_b_descriptor,
            cb_output_tiles_descriptor,
        ],
    )
