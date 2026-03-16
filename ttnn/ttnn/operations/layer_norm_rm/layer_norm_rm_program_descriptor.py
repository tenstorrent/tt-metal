# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Builds the ProgramDescriptor with:
  - 15 circular buffers (per op_design.md)
  - Reader, Compute, Writer kernel descriptors
  - Work distribution across an 8x8 core grid (tile-rows split across cores)
"""

import struct
from pathlib import Path
from math import prod

import ttnn


# ---------------------------------------------------------------------------
# Kernel file paths
# ---------------------------------------------------------------------------
KERNEL_DIR = Path(__file__).parent / "kernels"
READER_KERNEL = str(KERNEL_DIR / "reader_layer_norm_rm.cpp")
COMPUTE_KERNEL = str(KERNEL_DIR / "compute_layer_norm_rm.cpp")
# Writer: reuse existing layernorm RM writer from the codebase
WRITER_KERNEL = (
    "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_blocked_rm_output.cpp"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE_H = 32
TILE_W = 32
BF16_TILE_SIZE = 2048  # 32 * 32 * 2 bytes for bfloat16


def pack_bfloat16_pair(value: float) -> int:
    """Pack a float as two bfloat16 values into a uint32.

    The kernel expects (bf16 << 16 | bf16) for scalar broadcast.
    """
    fp32_bytes = struct.pack(">f", value)
    bf16 = int.from_bytes(fp32_bytes[:2], "big")
    return (bf16 << 16) | bf16


def _largest_divisor_le(n: int, limit: int) -> int:
    """Return the largest divisor of n that is <= limit."""
    for d in range(min(n, limit), 0, -1):
        if n % d == 0:
            return d
    return 1


def build_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Build the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor:  Input tensor on device (bfloat16, RM, interleaved).
        output_tensor: Pre-allocated output tensor on device.
        gamma:         Optional scale tensor (1,1,1,W), RM, bfloat16.
        beta:          Optional shift tensor (1,1,1,W), RM, bfloat16.
        epsilon:       Numerical stability constant.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op().
    """
    # ================================================================
    # 1. Extract tensor metadata
    # ================================================================
    shape = input_tensor.shape
    ndim = len(shape)
    W = shape[ndim - 1]
    Wt = W // TILE_W  # width in tiles

    # Total tile-rows = product of all dims except last / TILE_H
    total_rows = prod(shape[i] for i in range(ndim - 1)) if ndim > 1 else 1
    # For the H dimension we need total_rows (already the product of all non-W dims)
    # but we also need to account that H is tile-aligned; tile-rows = total_rows / 32
    num_tile_rows = total_rows // TILE_H

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    stick_size = W * 2  # bytes per RM row (bfloat16 = 2 bytes)
    block_size = _largest_divisor_le(Wt, 8)
    elem_size_bytes = 2  # bfloat16

    # ================================================================
    # 2. Core grid and work distribution
    # ================================================================
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(compute_grid.x, 8)
    grid_y = min(compute_grid.y, 8)
    grid_size = ttnn.CoreCoord(grid_x, grid_y)

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tile_rows_per_core_g1,
        tile_rows_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, num_tile_rows)

    # ================================================================
    # 3. Circular buffer descriptors (15 CBs)
    # ================================================================
    page_size = BF16_TILE_SIZE  # All CBs use tile-sized pages for bf16

    def make_cb(cb_index: int, num_pages: int) -> ttnn.CBDescriptor:
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_index,
                    data_format=ttnn.bfloat16,
                    page_size=page_size,
                )
            ],
        )

    cbs = [
        make_cb(0, Wt),  # c_0:  cb_in_rm     - RM input sticks
        make_cb(1, Wt),  # c_1:  cb_in         - Tilized input
        make_cb(2, 1),  # c_2:  cb_scaler     - Reduce scaler (1.0)
        make_cb(3, 1),  # c_3:  cb_mean       - Row-wise mean
        make_cb(4, Wt),  # c_4:  cb_centered   - x - mean
        make_cb(5, Wt),  # c_5:  cb_sq         - (x - mean)^2
        make_cb(6, 1),  # c_6:  cb_var        - Row-wise variance
        make_cb(7, 1),  # c_7:  cb_eps        - Epsilon constant
        make_cb(16, Wt),  # c_16: cb_out        - Final output tiles
        make_cb(17, Wt),  # c_17: cb_gamma      - Tilized gamma
        make_cb(18, Wt),  # c_18: cb_beta       - Tilized beta
        make_cb(19, Wt),  # c_19: cb_gamma_rm   - RM gamma sticks
        make_cb(20, Wt),  # c_20: cb_beta_rm    - RM beta sticks
        make_cb(24, 1),  # c_24: cb_rsqrt      - 1/sqrt(var+eps)
        make_cb(25, Wt),  # c_25: cb_temp       - Scratch for affine routing
        make_cb(28, Wt),  # c_28: cb_out_rm     - RM output for writer
    ]

    # ================================================================
    # 4. Kernel descriptors
    # ================================================================

    # --- 4a. Reader kernel ---
    # Compile-time args: stick_size, Wt, has_gamma, has_beta,
    #   TensorAccessorArgs(input), TensorAccessorArgs(gamma_or_placeholder),
    #   TensorAccessorArgs(beta_or_placeholder)
    reader_ct_args = [stick_size, Wt, has_gamma, has_beta]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # For gamma/beta: always provide TensorAccessorArgs slot (2 values for interleaved).
    # When absent, use input tensor's accessor as placeholder (runtime addr=0 prevents reads).
    if gamma is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if beta is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Reader runtime args (per core)
    scaler_packed = pack_bfloat16_pair(1.0)
    eps_packed = pack_bfloat16_pair(epsilon)

    reader_rt_args = ttnn.RuntimeArgs()
    current_tile_row = 0

    def _set_reader_rt(x, y, num_rows, start_row):
        reader_rt_args[x][y] = [
            input_tensor.buffer_address(),
            num_rows,
            start_row,
            gamma.buffer_address() if gamma is not None else 0,
            beta.buffer_address() if beta is not None else 0,
            scaler_packed,
            eps_packed,
        ]

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                _set_reader_rt(x, y, tile_rows_per_core_g1, current_tile_row)
                current_tile_row += tile_rows_per_core_g1

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                _set_reader_rt(x, y, tile_rows_per_core_g2, current_tile_row)
                current_tile_row += tile_rows_per_core_g2

    # Set empty runtime args for idle cores in the grid
    _set_idle_cores(reader_rt_args, grid_x, grid_y, all_cores)

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- 4b. Compute kernel ---
    compute_ct_args = [Wt, block_size, has_gamma, has_beta]

    compute_rt_args = ttnn.RuntimeArgs()
    current_tile_row = 0

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                compute_rt_args[x][y] = [tile_rows_per_core_g1, W]
                current_tile_row += tile_rows_per_core_g1

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                compute_rt_args[x][y] = [tile_rows_per_core_g2, W]
                current_tile_row += tile_rows_per_core_g2

    _set_idle_cores(compute_rt_args, grid_x, grid_y, all_cores)

    compute_defines = [("TILIZE_IN", "1"), ("UNTILIZE_OUT", "1")]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL,
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        defines=compute_defines,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- 4c. Writer kernel ---
    # Writer compile-time args: block_size, TensorAccessorArgs(output), elem_size_bytes
    writer_ct_args = [block_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.append(elem_size_bytes)

    # H_logical = total valid rows (product of all dims except last)
    H_logical = total_rows

    writer_rt_args = ttnn.RuntimeArgs()
    current_tile_row = 0

    def _set_writer_rt(x, y, num_rows, start_row):
        writer_rt_args[x][y] = [
            output_tensor.buffer_address(),
            Wt,
            num_rows,
            start_row,
            H_logical,
        ]

    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                _set_writer_rt(x, y, tile_rows_per_core_g1, current_tile_row)
                current_tile_row += tile_rows_per_core_g1

    for core_range in core_group_2.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                _set_writer_rt(x, y, tile_rows_per_core_g2, current_tile_row)
                current_tile_row += tile_rows_per_core_g2

    _set_idle_cores(writer_rt_args, grid_x, grid_y, all_cores)

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ================================================================
    # 5. Assemble ProgramDescriptor
    # ================================================================
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )


def _set_idle_cores(
    rt_args: ttnn.RuntimeArgs,
    grid_x: int,
    grid_y: int,
    active_cores: ttnn.CoreRangeSet,
) -> None:
    """Set empty runtime args for all idle cores in the grid.

    CRITICAL: Every core in the grid must have runtime args set (even if empty).
    """
    active_set = set()
    for core_range in active_cores.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                active_set.add((x, y))

    for x in range(grid_x):
        for y in range(grid_y):
            if (x, y) not in active_set:
                rt_args[x][y] = []
