# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
row_mean_rm - Program Descriptor

CB layout:
  c_0  (cb_input_rm)     - Input RM sticks (Wt pages)
  c_8  (cb_scaler)       - Reduce scaler 1/W, 1 tile, never popped
  c_16 (cb_out_rm)       - Output RM sticks (1 page — single tile column)
  c_24 (cb_input_tiled)  - Tilized input (Wt tiles)
  c_25 (cb_mean)         - Mean tile (1 tile)
"""

import struct
from pathlib import Path
import ttnn

_OP_DIR = Path(__file__).parent
KERNEL_DIR = _OP_DIR / "kernels"

_READER_KERNEL = str(KERNEL_DIR / "row_mean_rm_reader.cpp")
_COMPUTE_KERNEL = str(KERNEL_DIR / "row_mean_rm_compute.cpp")
_WRITER_KERNEL = str(KERNEL_DIR / "row_mean_rm_writer.cpp")

CB_INPUT_RM = 0
CB_SCALER = 8
CB_OUT_RM = 16
CB_INPUT_TILED = 24
CB_MEAN = 25


def _float_to_uint32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    rank = len(input_tensor.shape)
    W = input_tensor.shape[rank - 1]
    H = input_tensor.shape[rank - 2]

    N_outer = 1
    for i in range(rank - 2):
        N_outer *= input_tensor.shape[i]

    Wt = W // 32
    Ht = H // 32
    num_rows = N_outer * Ht

    tile_size = ttnn.tile_size(input_tensor.dtype)
    element_size = tile_size // (32 * 32)
    input_stick_size = W * element_size
    output_stick_size = 32 * element_size  # output is 32 wide

    rm_cb_page_size = tile_size

    # Single-core grid
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Circular buffers
    cb_input_rm = ttnn.CBDescriptor(
        total_size=Wt * rm_cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_RM,
                data_format=input_tensor.dtype,
                page_size=rm_cb_page_size,
            )
        ],
    )

    cb_scaler = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # Output CB: 1 tile (single tile column)
    cb_out_rm = ttnn.CBDescriptor(
        total_size=1 * rm_cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT_RM,
                data_format=output_tensor.dtype,
                page_size=rm_cb_page_size,
            )
        ],
    )

    cb_input_tiled = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_mean = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    all_cbs = [cb_input_rm, cb_scaler, cb_out_rm, cb_input_tiled, cb_mean]

    # Reader kernel
    reader_ct_args = [input_stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    scaler_value = _float_to_uint32_bits(1.0 / W)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_rows,
        Wt,
        0,  # start_stick_id
        scaler_value,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Writer kernel
    writer_ct_args = [output_stick_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_rows,
        0,  # start_stick_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        core_ranges=core_grid,
        compile_time_args=[num_rows, Wt],
        defines=[],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
