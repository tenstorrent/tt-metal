# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for toy_binary_in_place.

Supports two modes:
  in_place=True:  copy A→cb_work, add_in_place(cb_work, cb_b), copy cb_work→cb_out
  in_place=False: add(cb_input, cb_b, cb_out) — standard non-in-place binary add

CB layout:
  CB_INPUT (0)  — Input A tiles from reader, double-buffered.
  CB_B     (1)  — Input B tiles from reader, sized per broadcast mode.
  CB_WORK  (2)  — Compute-only working buffer (in-place mode only), Ht*Wt tiles.
  CB_OUT   (16) — Output tiles, double-buffered. Compute → Writer.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

BCAST_MODE_MAP = {"none": 0, "row": 1, "col": 2, "scalar": 3}
OP_CODE_MAP = {"add": 0, "sub": 1, "mul": 2, "square": 3, "sfpu_square": 4}


def create_program_descriptor(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    broadcast_mode: str = "none",
    in_place: bool = True,
    op: str = "add",
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
) -> ttnn.ProgramDescriptor:
    a_shape = list(input_a.shape)
    Wt = (a_shape[-1] + TILE_DIM - 1) // TILE_DIM
    Ht = (a_shape[-2] + TILE_DIM - 1) // TILE_DIM
    total_a_tiles = Ht * Wt

    bcast_code = BCAST_MODE_MAP[broadcast_mode]
    in_place_flag = 1 if in_place else 0
    op_code = OP_CODE_MAP[op]

    # B tile count depends on broadcast mode
    if broadcast_mode == "none":
        b_tiles = Ht * Wt
    elif broadcast_mode == "row":
        b_tiles = Wt
    elif broadcast_mode == "col":
        b_tiles = Ht
    else:  # scalar
        b_tiles = 1

    a_page_size = input_a.buffer_page_size()
    a_num_pages = input_a.buffer_num_pages()
    b_page_size = input_b.buffer_page_size()
    b_num_pages = input_b.buffer_num_pages()
    out_page_size = output_tensor.buffer_page_size()
    out_num_pages = output_tensor.buffer_num_pages()

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Circular Buffers ---
    CB_INPUT = 0
    CB_B = 1
    CB_WORK = 2
    CB_OUT = 16

    cbs = [
        # CB_INPUT: double-buffered, reader → compute streaming
        ttnn.CBDescriptor(
            total_size=2 * a_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_INPUT, data_format=input_a.dtype, page_size=a_page_size)
            ],
        ),
        # CB_B: all B tiles + 1 for double-buffering
        ttnn.CBDescriptor(
            total_size=(b_tiles + 1) * b_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_B, data_format=input_b.dtype, page_size=b_page_size)
            ],
        ),
        # CB_WORK: all A tiles — compute-only, in-place target (only used when in_place=True)
        ttnn.CBDescriptor(
            total_size=max(1, total_a_tiles if in_place else 1) * a_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_WORK, data_format=input_a.dtype, page_size=a_page_size)
            ],
        ),
        # CB_OUT: double-buffered for streaming to writer
        ttnn.CBDescriptor(
            total_size=2 * out_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=out_page_size)
            ],
        ),
    ]

    # --- Reader ---
    reader_ct_args = [a_num_pages, b_num_pages]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_b).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_a.buffer_address(),
        0,
        input_b.buffer_address(),
        0,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = [out_num_pages]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), 0]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    compute_ct_args = [Ht, Wt, bcast_code, in_place_flag, op_code]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
