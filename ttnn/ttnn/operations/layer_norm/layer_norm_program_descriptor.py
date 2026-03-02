# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm - Program Descriptor

Defines CBs, kernels, and runtime args for the layer_norm generic_op.

Architecture:
- Single core (0, 0)
- Reader: loads input rows, gamma/beta (optional), fills scaler and eps CBs
- Compute: 3-pass normalization per row (mean, variance, normalize + optional affine)
- Writer: drains output tiles to DRAM
"""

import struct
from pathlib import Path
import ttnn

# Kernel files are in the kernels/ subdirectory relative to this file
KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (must match kernel code)
CB_INPUT = 0  # c_0:  input tiles (Wt per row, persistent)
CB_SCALER = 1  # c_1:  reduce scaler (1/Wt), 1 tile
CB_EPS = 2  # c_2:  epsilon, 1 tile
CB_GAMMA = 3  # c_3:  gamma tiles (Wt per row, if present)
CB_BETA = 4  # c_4:  beta tiles (Wt per row, if present)
CB_OUT = 16  # c_16: output tiles (Wt per row)
CB_MEAN = 24  # c_24: mean per row (1 tile)
CB_CENTERED = 25  # c_25: x - mean (Wt tiles, persistent)
CB_SQUARED = 26  # c_26: (x - mean)^2 tiles (Wt tiles)
CB_VAR = 27  # c_27: variance per row (1 tile)
CB_RSTD = 28  # c_28: 1/sqrt(var+eps) per row (1 tile)
CB_NORMALIZED = 29  # c_29: normalized tiles before affine (Wt tiles, if affine)
CB_GAMMA_OUT = 30  # c_30: after gamma multiply (Wt tiles, if gamma+beta)


def _pack_float_as_uint32(val: float) -> int:
    """Pack a Python float as a uint32 (IEEE 754 bit representation)."""
    return struct.unpack("I", struct.pack("f", val))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    weight: ttnn.Tensor = None,
    bias: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the layer_norm operation.

    Args:
        input_tensor:  Input tensor on device (TILE_LAYOUT, bfloat16, 4D)
        output_tensor: Pre-allocated output tensor on device
        weight:        Optional gamma tensor
        bias:          Optional beta tensor
        epsilon:       Numerical stability constant

    Returns:
        ProgramDescriptor ready for ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    # Shape is [N, C, H, W]; all dims guaranteed multiples of 32 (tile-aligned)
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    Wt = W // 32  # width in tiles
    Ht = H // 32  # height in tiles per NC slice
    num_rows = N * C * Ht  # total tile-rows to process

    gamma_has_value = 1 if weight is not None else 0
    beta_has_value = 1 if bias is not None else 0

    # Tile size in bytes for bfloat16 (2 bytes per element, 32x32 = 2048 bytes)
    tile_size = input_tensor.tile.get_tile_size(input_tensor.dtype)

    # ========== 2. CORE GRID (single core) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    dtype = input_tensor.dtype

    def make_cb(cb_id: int, num_pages: int) -> ttnn.CBDescriptor:
        """Helper: create a tile-based CB with given id and page count."""
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_id,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )

    # Input CB: holds Wt tiles (one full row), persistent across 3 compute passes
    cb_input_desc = make_cb(CB_INPUT, Wt)

    # Scaler CB: 1 tile (reduce scaler = 1/Wt for REDUCE_ROW AVG)
    cb_scaler_desc = make_cb(CB_SCALER, 1)

    # Epsilon CB: 1 tile
    cb_eps_desc = make_cb(CB_EPS, 1)

    # Gamma CB: Wt tiles if present, else 1 tile (placeholder, not read)
    cb_gamma_pages = Wt if gamma_has_value else 1
    cb_gamma_desc = make_cb(CB_GAMMA, cb_gamma_pages)

    # Beta CB: Wt tiles if present, else 1 tile (placeholder, not read)
    cb_beta_pages = Wt if beta_has_value else 1
    cb_beta_desc = make_cb(CB_BETA, cb_beta_pages)

    # Output CB: Wt tiles per row
    cb_out_desc = make_cb(CB_OUT, Wt)

    # Intermediate CBs
    cb_mean_desc = make_cb(CB_MEAN, 1)
    cb_centered_desc = make_cb(CB_CENTERED, Wt)
    cb_squared_desc = make_cb(CB_SQUARED, Wt)
    cb_var_desc = make_cb(CB_VAR, 1)
    cb_rstd_desc = make_cb(CB_RSTD, 1)

    # Normalized CB: needed only if affine transform is applied
    cb_normalized_pages = Wt if (gamma_has_value or beta_has_value) else 1
    cb_normalized_desc = make_cb(CB_NORMALIZED, cb_normalized_pages)

    # Gamma-out CB: needed only if both gamma and beta present
    cb_gamma_out_pages = Wt if (gamma_has_value and beta_has_value) else 1
    cb_gamma_out_desc = make_cb(CB_GAMMA_OUT, cb_gamma_out_pages)

    all_cbs = [
        cb_input_desc,
        cb_scaler_desc,
        cb_eps_desc,
        cb_gamma_desc,
        cb_beta_desc,
        cb_out_desc,
        cb_mean_desc,
        cb_centered_desc,
        cb_squared_desc,
        cb_var_desc,
        cb_rstd_desc,
        cb_normalized_desc,
        cb_gamma_out_desc,
    ]

    # ========== 4. READER KERNEL ==========
    # Compile-time args: gamma_has_value, beta_has_value, TensorAccessorArgs(input),
    # [TensorAccessorArgs(gamma) if present], [TensorAccessorArgs(beta) if present]
    reader_ct_args = [gamma_has_value, beta_has_value]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if weight is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    if bias is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(bias).get_compile_time_args())

    # Runtime args: input_addr, gamma_addr (0 if none), beta_addr (0 if none),
    #               num_rows, Wt, start_tile_id=0, epsilon_as_uint32
    gamma_addr = weight.buffer_address() if weight is not None else 0
    beta_addr = bias.buffer_address() if bias is not None else 0
    eps_u32 = _pack_float_as_uint32(float(epsilon))

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_addr,
        beta_addr,
        num_rows,
        Wt,
        0,  # start_tile_id
        eps_u32,  # epsilon packed as uint32 (IEEE 754 float bits)
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_layer_norm.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ========== 5. COMPUTE KERNEL ==========
    # Compile-time args: Wt, num_rows, gamma_has_value, beta_has_value
    compute_ct_args = [Wt, num_rows, gamma_has_value, beta_has_value]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_layer_norm.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 6. WRITER KERNEL ==========
    # Compile-time args: TensorAccessorArgs(output)
    writer_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    # Runtime args: output_addr, num_rows, Wt, start_tile_id=0
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_rows,
        Wt,
        0,  # start_tile_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_layer_norm.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 7. ASSEMBLE PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
