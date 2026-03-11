# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines ProgramDescriptor: circular buffers, kernel descriptors, and runtime args
for layer normalization on row-major interleaved tensors.

Single-core operation on core (0,0).

CB Layout:
    c_0  (cb_in)         - RM sticks from reader -> tilize in compute (Wt pages)
    c_1  (cb_tilized)    - Tilized input tiles (Wt pages)
    c_2  (cb_mean)       - Row mean from reduce_row (1 page)
    c_3  (cb_centered)   - x - mean (Wt pages)
    c_4  (cb_sq)         - (x - mean)^2 (Wt pages)
    c_5  (cb_var)        - Row variance from reduce_row (1 page)
    c_6  (cb_eps)        - Epsilon constant tile (1 page)
    c_7  (cb_inv_std)    - 1/sqrt(var+eps) (1 page)
    c_8  (cb_scaler)     - Reduce scaler 1/W (1 page)
    c_9  (cb_gamma)      - Gamma tiles (Wt pages, program lifetime)
    c_10 (cb_beta)       - Beta tiles (Wt pages, program lifetime)
    c_24 (cb_normalized) - Normalized output tiles (Wt pages)
    c_25 (cb_affine_tmp) - Intermediate for gamma*normalize when both gamma/beta (Wt pages)
    c_16 (cb_out)        - Untilized RM output for writer (Wt pages)
"""

import struct
from pathlib import Path

import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def _pack_bfloat16_pair(value: float) -> int:
    """Pack a float as two bfloat16 values into a uint32.

    This is used for passing epsilon and other constants as runtime args.
    The reader kernel uses fill_with_val_bfloat16 which expects this format.
    """
    f32_bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bf16_bits = f32_bits >> 16
    packed = (bf16_bits << 16) | bf16_bits
    return packed


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor: Input tensor (ROW_MAJOR, bfloat16, interleaved)
        output_tensor: Pre-allocated output tensor (ROW_MAJOR, bfloat16, interleaved)
        gamma: Optional per-feature scale tensor (1,1,1,W) ROW_MAJOR bfloat16
        beta: Optional per-feature shift tensor (1,1,1,W) ROW_MAJOR bfloat16
        epsilon: Numerical stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    W = shape[-1]
    # Compute total height by flattening all dims except the last
    total_height = 1
    for i in range(len(shape) - 1):
        total_height *= shape[i]

    Wt = W // 32  # tiles per row
    Ht = total_height // 32  # total tile-rows

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # Tile size for bfloat16 (32x32 tiles = 2048 bytes with bf16)
    tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Stick size for RM tensors: W elements * 2 bytes per bf16
    stick_size = W * 2

    # ========== 2. CORE GRID (single core) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices
    CB_IN = 0
    CB_TILIZED = 1
    CB_MEAN = 2
    CB_CENTERED = 3
    CB_SQ = 4
    CB_VAR = 5
    CB_EPS = 6
    CB_INV_STD = 7
    CB_SCALER = 8
    CB_GAMMA = 9
    CB_BETA = 10
    CB_OUT = 16
    CB_NORMALIZED = 24
    CB_AFFINE_TMP = 25

    dtype = input_tensor.dtype  # bfloat16

    cb_descriptors = []

    # c_0: cb_in - RM sticks from reader (Wt pages of tile_size)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_IN,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_1: cb_tilized - Tilized input tiles (Wt pages)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_TILIZED,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_2: cb_mean - Row mean (1 page)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MEAN,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_3: cb_centered - x - mean (Wt pages)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_4: cb_sq - (x - mean)^2 (Wt pages)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SQ,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_5: cb_var - Row variance (1 page)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_VAR,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_6: cb_eps - Epsilon constant (1 page, program lifetime)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EPS,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_7: cb_inv_std - 1/sqrt(var+eps) (1 page)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INV_STD,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_8: cb_scaler - Reduce scaler 1/W (1 page, program lifetime)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_9: cb_gamma - Gamma tiles (Wt pages, program lifetime)
    if has_gamma:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_10: cb_beta - Beta tiles (Wt pages, program lifetime)
    if has_beta:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_24: cb_normalized - Normalized output tiles (Wt pages)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_NORMALIZED,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_25: cb_affine_tmp - Intermediate for gamma result when both gamma/beta (Wt pages)
    if has_gamma:
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_AFFINE_TMP,
                        data_format=dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # c_16: cb_out - Untilized RM output for writer (Wt pages)
    cb_descriptors.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [
        stick_size,  # 0: stick_size (W * 2 bytes)
        Wt,  # 1: tiles per row
        Ht,  # 2: total tile-rows
        W,  # 3: width in elements
        has_gamma,  # 4: has_gamma flag
        has_beta,  # 5: has_beta flag
    ]
    # Append TensorAccessor args for input
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Append TensorAccessor args for gamma if present
    if gamma is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    # Append TensorAccessor args for beta if present
    if beta is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    rt_list = [
        input_tensor.buffer_address(),  # 0: src_addr
        gamma.buffer_address() if gamma is not None else 0,  # 1: gamma_addr
        beta.buffer_address() if beta is not None else 0,  # 2: beta_addr
        _pack_bfloat16_pair(epsilon),  # 3: packed_eps (bf16 pair)
    ]
    reader_rt_args[core.x][core.y] = rt_list

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [
        Wt,  # 0: tiles per row
        Ht,  # 1: total tile-rows
        has_gamma,  # 2: has_gamma flag
        has_beta,  # 3: has_beta flag
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_layer_norm_rm.cpp"),
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
    writer_ct_args = [
        stick_size,  # 0: stick_size (W * 2 bytes)
        Wt,  # 1: tiles per row
        Ht,  # 2: total tile-rows
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),  # 0: dst_addr
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_layer_norm_rm.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )
