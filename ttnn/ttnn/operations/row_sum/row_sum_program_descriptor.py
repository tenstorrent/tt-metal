# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    fp32_dest_acc_en: bool = False,
) -> ttnn.ProgramDescriptor:
    """
    Create ProgramDescriptor for row_sum: reduce each row via SUM along W dimension.

    Supports all 4 layout combinations (tiled/row-major input × tiled/row-major output).
    """
    # ========== 1. TENSOR METADATA ==========
    dtype = input_tensor.dtype
    tile_size = ttnn.tile_size(dtype)

    input_is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    output_is_rm = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    H, W = input_tensor.shape
    # Padded dimensions (tile-aligned)
    Ht = ttnn.div_up(H, 32)
    Wt = ttnn.div_up(W, 32)

    # For row-major input: stick bytes (actual width, not padded)
    input_row_bytes = W * input_tensor.element_size()
    # For row-major output: stick bytes (output width is 32)
    output_row_bytes = 32 * output_tensor.element_size()

    # Page counts
    if input_is_rm:
        input_num_pages = H  # sticks
    else:
        input_num_pages = input_tensor.buffer_num_pages()  # tiles: Ht * Wt

    if output_is_rm:
        output_num_pages = H  # sticks
    else:
        output_num_pages = output_tensor.buffer_num_pages()  # tiles: Ht * 1

    # ========== 2. CORE GRID ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFERS ==========
    # CB indices
    cb_in_idx = 0
    cb_scaler_idx = 1
    cb_out_idx = 16
    cb_tilized_idx = 24
    cb_reduced_idx = 25

    cbs = []

    # CB0: Input
    if input_is_rm:
        # Symmetric tilize mode: Wt tile-sized pages hold 32 sticks
        cb_in_pages = Wt
    else:
        # Tiled streaming: double buffer
        cb_in_pages = 2

    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb_in_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_in_idx,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # CB1: Scaler (1.0 for SUM)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_scaler_idx,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # CB16: Output
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_out_idx,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # CB24: Tilized intermediate (only if input is RM)
    if input_is_rm:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_tilized_idx,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # CB25: Reduced intermediate (only if output is RM)
    if output_is_rm:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_reduced_idx,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    # CT args: input_is_rm, cb_in, cb_scaler, num_input_tiles_or_H, Wt_or_row_bytes, + TensorAccessorArgs
    if input_is_rm:
        reader_ct_args = [1, cb_in_idx, cb_scaler_idx, H, input_row_bytes]
    else:
        num_input_tiles = Ht * Wt
        reader_ct_args = [0, cb_in_idx, cb_scaler_idx, num_input_tiles, 0]

    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        0,  # start_page_id
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_sum_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # CT args: output_is_rm, cb_out, num_output_tiles_or_H, output_row_bytes, + TensorAccessorArgs
    if output_is_rm:
        writer_ct_args = [1, cb_out_idx, H, output_row_bytes]
    else:
        num_output_tiles = Ht
        writer_ct_args = [0, cb_out_idx, num_output_tiles, 0]

    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        output_num_pages,
        0,  # start_page_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_sum_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    # CT args: Ht, Wt, input_is_rm, output_is_rm
    compute_ct_args = [Ht, Wt, int(input_is_rm), int(output_is_rm)]

    fp32_dest = fp32_dest_acc_en
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "row_sum_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest),
    )

    # ========== 5. RETURN ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
