# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm - Program Descriptor

Defines circular buffers, kernel descriptors, and runtime args for layer_norm.

Architecture (single-core 1x1):
  - Input: [1, 1, H, W] TILE BFP16  --> Ht * Wt tiles
  - Optional gamma: [1, 1, 1, W] TILE BFP16  --> Wt tiles
  - Optional beta:  [1, 1, 1, W] TILE BFP16  --> Wt tiles
  - Output: [1, 1, H, W] TILE BFP16 --> Ht * Wt tiles

CB layout:
  c_0  (cb_input)          : Wt pages, per-row input
  c_1  (cb_scaler)         : 1 page, 1/W reduce scaler (program lifetime)
  c_2  (cb_eps)            : 1 page, eps constant (program lifetime)
  c_3  (cb_gamma)          : Wt pages, gamma tiles (program lifetime)
  c_4  (cb_beta)           : Wt pages, beta tiles (program lifetime)
  c_16 (cb_output)         : Wt pages, per-row output
  c_24 (cb_mean / cb_rstd) : 1 page, dual-use: mean (P1-P2) / rstd (P4b-P5)
  c_25 (cb_x_minus_mean)   : Wt pages, x - mean intermediate
  c_26 (cb_variance)       : 1 page, intermediate variance
  c_27 (cb_diff_sq / cb_temp_norm): Wt pages, dual-use: (x-mean)^2 / temp normalized
"""

import struct
from pathlib import Path
import ttnn

# Kernel files are co-located with this module in kernels/ subdirectory.
# Paths passed to KernelDescriptor must be relative to the tt-metal repo root.
_THIS_DIR = Path(__file__).parent
_REPO_ROOT = _THIS_DIR
# Walk up to find repo root (contains tt_metal/ directory)
for _ in range(10):
    if (_REPO_ROOT / "tt_metal").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

KERNEL_DIR_RELATIVE = str(_THIS_DIR.relative_to(_REPO_ROOT) / "kernels")


def _float_to_uint32(f: float) -> int:
    """Reinterpret a float as its uint32 bit pattern."""
    return struct.unpack("I", struct.pack("f", f))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm.

    Args:
        input_tensor:  [1, 1, H, W] TILE BFP16, on device
        output_tensor: [1, 1, H, W] TILE BFP16, on device (pre-allocated)
        gamma:         [1, 1, 1, W] TILE BFP16, on device (optional)
        beta:          [1, 1, 1, W] TILE BFP16, on device (optional)
        eps:           Float epsilon for numerical stability

    Returns:
        ProgramDescriptor for ttnn.generic_op
    """
    # =========================================================
    # 1. TENSOR METADATA
    # =========================================================
    shape = input_tensor.shape
    H = shape[2]
    W = shape[3]
    Ht = H // 32  # height in tiles
    Wt = W // 32  # width in tiles

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    tile_size = input_tensor.tile.get_tile_size(input_tensor.dtype)  # bytes per tile
    dtype = input_tensor.dtype  # e.g. ttnn.bfloat16

    # =========================================================
    # 2. SINGLE-CORE GRID
    # =========================================================
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # =========================================================
    # 3. CIRCULAR BUFFER DESCRIPTORS
    # =========================================================
    # c_0: input -- Wt tiles per row, double-buffered for throughput
    cb_input = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_1: scaler (1/W) -- 1 tile, program lifetime
    cb_scaler = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_2: eps constant -- 1 tile, program lifetime
    cb_eps = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_3: gamma tiles -- Wt tiles, program lifetime
    cb_gamma = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=3,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_4: beta tiles -- Wt tiles, program lifetime
    cb_beta = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=4,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_16: output -- Wt tiles per row
    cb_output = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=16,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_24: mean / rstd -- 1 tile dual-use intermediate
    cb_mean_rstd = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=24,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_25: x - mean -- Wt tiles per row
    cb_x_minus_mean = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=25,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_26: variance -- 1 tile intermediate
    cb_variance = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=26,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_27: diff_sq / temp_norm -- Wt tiles dual-use intermediate
    cb_diff_sq_temp = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=27,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    all_cbs = [
        cb_input,
        cb_scaler,
        cb_eps,
        cb_gamma,
        cb_beta,
        cb_output,
        cb_mean_rstd,
        cb_x_minus_mean,
        cb_variance,
        cb_diff_sq_temp,
    ]

    # =========================================================
    # 4. KERNEL DESCRIPTORS
    # =========================================================

    # --- Reader ---
    # Compile-time: [Ht, Wt, has_gamma, has_beta] + TensorAccessorArgs(input)
    reader_ct_args = [Ht, Wt, has_gamma, has_beta]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Runtime: [input_addr, gamma_addr, beta_addr, eps_u32]
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0
    eps_u32 = _float_to_uint32(eps)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_addr,
        beta_addr,
        eps_u32,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR_RELATIVE}/layer_norm_reader.cpp",
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    # Compile-time: [Ht, Wt] + TensorAccessorArgs(output)
    writer_ct_args = [Ht, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Runtime: [output_addr]
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR_RELATIVE}/layer_norm_writer.cpp",
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    # Compile-time: [Ht, Wt, has_gamma, has_beta]
    compute_ct_args = [Ht, Wt, has_gamma, has_beta]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR_RELATIVE}/layer_norm_compute.cpp",
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # =========================================================
    # 5. PROGRAM DESCRIPTOR
    # =========================================================
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
