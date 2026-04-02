# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for toy_rm_row_mean.

This keeps the layer_norm_rm handoff shape that matters for debugging:
ROW_MAJOR reader -> tilize -> REDUCE_ROW -> tiled writer.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_uint32(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    compute_kernel_config: dict = None,
    post_tilize_nops: int = 0,
    insert_tensix_sync: bool = False,
) -> ttnn.ProgramDescriptor:
    shape = list(input_tensor.shape)
    width = shape[-1]
    height = shape[-2]
    nc = math.prod(shape[:-2]) if len(shape) > 2 else 1

    width_tiles = width // 32
    height_tiles = height // 32
    num_blocks = nc * height_tiles
    total_num_rows = nc * height

    row_bytes = width * input_tensor.element_size()
    tile_size = ttnn.tile_size(input_tensor.dtype)
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)
    output_num_tiles = output_tensor.buffer_num_pages()

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cb_rm_in = 0
    cb_scaler = 8
    cb_out = 16
    cb_x = 24

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * width_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_rm_in,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_scaler,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_out,
                    data_format=output_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=width_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_x,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        ),
    ]

    scaler_bits = _float_to_uint32(1.0 / width)

    reader_ct_args = [row_bytes]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        scaler_bits,
        total_num_rows,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct_args = [output_num_tiles]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        0,
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct_args = [width_tiles, num_blocks, post_tilize_nops, 1 if insert_tensix_sync else 0]

    compute_config_kwargs = {}
    if compute_kernel_config is not None:
        if "math_fidelity" in compute_kernel_config:
            compute_config_kwargs["math_fidelity"] = compute_kernel_config["math_fidelity"]
        if "fp32_dest_acc_en" in compute_kernel_config:
            compute_config_kwargs["fp32_dest_acc_en"] = compute_kernel_config["fp32_dest_acc_en"]
        if "math_approx_mode" in compute_kernel_config:
            compute_config_kwargs["math_approx_mode"] = compute_kernel_config["math_approx_mode"]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(**compute_config_kwargs),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
