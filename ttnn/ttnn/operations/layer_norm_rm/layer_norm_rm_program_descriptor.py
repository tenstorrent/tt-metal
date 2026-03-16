# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layer Norm RM - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for per-row layer normalization on RM interleaved tensors.

CB layout:
    0  cb_in         - RM sticks for tilize (Wt pages)
    1  cb_gamma       - Tilized gamma tiles (Wt pages, entire kernel)
    2  cb_beta        - Tilized beta tiles (Wt pages, entire kernel)
    3  cb_gamma_rm    - RM gamma sticks for tilize (Wt pages, loaded once)
    4  cb_beta_rm     - RM beta sticks for tilize (Wt pages, loaded once)
    8  cb_scaler      - Reduce scaler 1/W (1 page, entire kernel)
    9  cb_eps         - Epsilon scalar tile (1 page, entire kernel)
    16 cb_out         - RM sticks from untilize (Wt pages)
    24 cb_tilized     - Tilized input tiles (Wt pages)
    25 cb_mean        - Row means (1 page)
    26 cb_centered    - Centered tiles (Wt pages)
    27 cb_sq          - Squared centered tiles (Wt pages)
    28 cb_var         - Variance tile (1 page)
    29 cb_inv_std     - 1/sqrt(var+eps) (1 page)
    30 cb_norm        - Normalized tiles (Wt pages)
"""

import struct
from pathlib import Path
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def float_to_packed_bf16(val: float) -> int:
    """Convert float to packed bf16 format (bf16 << 16 | bf16).

    This is the format expected by prepare_reduce_scaler on device.
    """
    f32_bytes = struct.pack(">f", val)
    bf16 = int.from_bytes(f32_bytes[:2], "big")
    return (bf16 << 16) | bf16


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor: Input tensor (on device, RM, bfloat16, interleaved DRAM)
        output_tensor: Pre-allocated output tensor (on device)
        gamma: Optional scale tensor (on device)
        beta: Optional bias tensor (on device)
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    W = shape[-1]
    # H = product of all dimensions except the last
    H = 1
    for i in range(len(shape) - 1):
        H *= shape[i]

    Wt = W // 32  # tiles per row
    Ht = H // 32  # total tile-rows to process

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # Tile size for bf16 32x32 tiles = 2048 bytes
    tile_size = ttnn.tile_size(ttnn.bfloat16)

    # ========== 2. CORE GRID (single core) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices
    CB_IN = 0
    CB_GAMMA = 1
    CB_BETA = 2
    CB_GAMMA_RM = 3
    CB_BETA_RM = 4
    CB_SCALER = 8
    CB_EPS = 9
    CB_OUT = 16
    CB_TILIZED = 24
    CB_MEAN = 25
    CB_CENTERED = 26
    CB_SQ = 27
    CB_VAR = 28
    CB_INV_STD = 29
    CB_NORM = 30

    dtype = input_tensor.dtype  # bfloat16

    def make_cb(cb_index, num_pages, page_size=tile_size):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_index,
                    data_format=dtype,
                    page_size=page_size,
                )
            ],
        )

    cbs = [
        # Input / output RM data
        make_cb(CB_IN, Wt),  # RM sticks for tilize
        make_cb(CB_OUT, Wt),  # RM sticks from untilize
        # Scalers (1 tile each, entire kernel)
        make_cb(CB_SCALER, 1),  # 1/W reduce scaler
        make_cb(CB_EPS, 1),  # epsilon scalar
        # Intermediate compute CBs
        make_cb(CB_TILIZED, Wt),  # tilized input
        make_cb(CB_MEAN, 1),  # row means
        make_cb(CB_CENTERED, Wt),  # x - mean
        make_cb(CB_SQ, Wt),  # (x - mean)^2
        make_cb(CB_VAR, 1),  # row variance
        make_cb(CB_INV_STD, 1),  # 1/sqrt(var+eps)
        make_cb(CB_NORM, Wt),  # normalized output
    ]

    # Conditional gamma/beta CBs
    if has_gamma:
        cbs.append(make_cb(CB_GAMMA, Wt))  # tilized gamma
        cbs.append(make_cb(CB_GAMMA_RM, Wt))  # RM gamma for tilize
    if has_beta:
        cbs.append(make_cb(CB_BETA, Wt))  # tilized beta
        cbs.append(make_cb(CB_BETA_RM, Wt))  # RM beta for tilize

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [Wt, Ht, has_gamma, has_beta]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Optional gamma tensor accessor (always occupy a slot)
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.extend([0])
    # Optional beta tensor accessor (always occupy a slot)
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    else:
        reader_ct_args.extend([0])

    # Pack scaler and epsilon as bf16 pairs
    scaler_packed = float_to_packed_bf16(1.0 / W)
    eps_packed = float_to_packed_bf16(epsilon)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address() if has_gamma else 0,
        beta.buffer_address() if has_beta else 0,
        scaler_packed,
        eps_packed,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [Wt, Ht, has_gamma, has_beta]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [Wt, Ht]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )
