# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Template Op - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.
"""

from pathlib import Path
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the template operation.

    Args:
        input_tensor: Input tensor (on device)
        output_tensor: Pre-allocated output tensor (on device)

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    # Never hard-code dtype, tile size, or element size. Extract from tensors,
    # just like C++ program factories do (see upsample_program_factory_multicore_interleaved.cpp).
    #
    # Available metadata APIs:
    #   input_tensor.dtype                                  # DataType enum (bfloat16, float32, ...)
    #   input_tensor.layout                                 # Layout enum (TILE_LAYOUT, ROW_MAJOR_LAYOUT)
    #   input_tensor.element_size()                         # bytes per element (2 for bf16, 4 for f32)
    #   input_tensor.tile.tile_shape                        # (height, width), e.g. (32, 32)
    #   input_tensor.tile.get_tile_size(input_tensor.dtype) # full tile size in bytes (e.g. 2048 for bf16)
    #   input_tensor.padded_shape                           # shape with padding
    #   input_tensor.volume()                               # total element count
    #
    # For TILE_LAYOUT (this example):
    #   page_size  = tensor.tile.get_tile_size(tensor.dtype)   # one tile
    #   num_pages  = tensor.volume() // (tile_h * tile_w)      # number of tiles
    #
    # For ROW_MAJOR_LAYOUT:
    #   page_size  = tensor.padded_shape[-1] * tensor.element_size()   # one row (stick)
    #   num_pages  = tensor.volume() // tensor.padded_shape[-1]        # number of rows

    input_dtype = input_tensor.dtype
    output_dtype = output_tensor.dtype
    tile_height, tile_width = input_tensor.tile.tile_shape
    input_page_size = input_tensor.tile.get_tile_size(input_dtype)
    output_page_size = output_tensor.tile.get_tile_size(output_dtype)
    num_pages = input_tensor.volume() // (tile_height * tile_width)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Example: use single core for simplicity
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices: 0-7 for input, 16-23 for output, 24-31 for intermediate
    cb_in = 0
    cb_out = 16

    cb_in_descriptor = ttnn.CBDescriptor(
        total_size=2 * input_page_size,  # double buffer
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_in,
                data_format=input_dtype,
                page_size=input_page_size,
            )
        ],
    )

    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=2 * output_page_size,  # double buffer
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_out,
                data_format=output_dtype,
                page_size=output_page_size,
            )
        ],
    )

    # ========== 4. KERNEL DESCRIPTORS ==========
    # Reader kernel
    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_pages,
        0,  # start_page_id
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "template_op_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer kernel
    writer_ct_args = [cb_out]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_pages,
        0,  # start_page_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "template_op_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel
    compute_ct_args = [num_pages]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "template_op_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in_descriptor, cb_out_descriptor],
    )
