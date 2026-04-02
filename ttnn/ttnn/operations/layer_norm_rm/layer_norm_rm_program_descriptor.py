# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — Program Descriptor.

Defines circular buffers, kernels, and runtime args for the layer normalization
operation on row-major interleaved tensors. Supports multi-core distribution
by splitting tile-row blocks across available cores.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_uint32(f: float) -> int:
    """Reinterpret a float32 as a uint32 for passing as a runtime arg."""
    return struct.unpack("I", struct.pack("f", f))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
    compute_kernel_config: dict = None,
) -> ttnn.ProgramDescriptor:
    """Create the ProgramDescriptor for layer_norm_rm."""

    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = list(input_tensor.shape)
    W = shape[-1]
    H = shape[-2]
    NC = math.prod(shape[:-2]) if len(shape) > 2 else 1

    Wt = W // 32
    Ht = H // 32
    num_blocks = NC * Ht
    elem_size = input_tensor.element_size()
    row_bytes = W * elem_size

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    tile_size = ttnn.tile_size(input_tensor.dtype)
    bf16_tile_size = ttnn.tile_size(ttnn.bfloat16)

    # padded_row_bytes for gamma/beta (ROW granularity CB page)
    padded_row_bytes = Wt * 32 * elem_size  # = W * elem_size = row_bytes

    # ========== 2. CORE GRID (multi-core) ==========
    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, num_blocks)

    has_group_2 = len(core_group_2.ranges()) > 0

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices
    CB_RM_IN = 0
    CB_GAMMA_RM = 1
    CB_BETA_RM = 2
    CB_SCALER = 8
    CB_EPS = 9
    CB_RM_OUT = 16
    CB_X = 24
    CB_REDUCE = 25
    CB_CENTERED = 26
    CB_SQ = 27
    CB_INV_STD = 28
    CB_NORMED = 29
    CB_GAMMA_T = 30
    CB_BETA_T = 31

    dtype = input_tensor.dtype

    def make_cb(cb_id, num_pages, page_size, data_format=None):
        if data_format is None:
            data_format = dtype
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_id,
                    data_format=data_format,
                    page_size=page_size,
                )
            ],
        )

    cbs = []

    # Input RM sticks — double-buffered, TILE granularity pages
    cbs.append(make_cb(CB_RM_IN, 2 * Wt, tile_size))

    # Output RM sticks — double-buffered, TILE granularity pages
    cbs.append(make_cb(CB_RM_OUT, 2 * Wt, tile_size))

    # Reduce scaler — 1 tile, always bfloat16
    cbs.append(make_cb(CB_SCALER, 1, bf16_tile_size, ttnn.bfloat16))

    # Epsilon — 1 tile, always bfloat16
    cbs.append(make_cb(CB_EPS, 1, bf16_tile_size, ttnn.bfloat16))

    # Intermediate CBs — all tile_size pages
    cbs.append(make_cb(CB_X, Wt, tile_size))  # tilized input / reused for affine
    cbs.append(make_cb(CB_REDUCE, 1, tile_size))  # reduce output (mean/var)
    cbs.append(make_cb(CB_CENTERED, Wt, tile_size))  # x - mean
    cbs.append(make_cb(CB_SQ, Wt, tile_size))  # centered^2
    cbs.append(make_cb(CB_INV_STD, 1, tile_size))  # 1/sqrt(var+eps)
    cbs.append(make_cb(CB_NORMED, Wt, tile_size))  # normalized output

    # Gamma CBs (conditional)
    if has_gamma:
        cbs.append(make_cb(CB_GAMMA_RM, 1, padded_row_bytes))  # RM gamma (1 row)
        cbs.append(make_cb(CB_GAMMA_T, Wt, tile_size))  # tilized gamma

    # Beta CBs (conditional)
    if has_beta:
        cbs.append(make_cb(CB_BETA_RM, 1, padded_row_bytes))  # RM beta (1 row)
        cbs.append(make_cb(CB_BETA_T, Wt, tile_size))  # tilized beta

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Shared compile-time args ---
    scaler_value = 1.0 / W
    scaler_bits = _float_to_uint32(scaler_value)
    eps_bits = _float_to_uint32(epsilon)

    input_ta_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    gamma_ta_ct_args = (
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    beta_ta_ct_args = (
        ttnn.TensorAccessorArgs(beta).get_compile_time_args()
        if beta is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    output_ta_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    # --- Reader kernel ---
    # Compile-time: Wt, row_bytes, has_gamma, has_beta, <input_ta>, <gamma_ta>, <beta_ta>
    reader_ct_args = [Wt, row_bytes, has_gamma, has_beta]
    reader_ct_args.extend(input_ta_ct_args)
    reader_ct_args.extend(gamma_ta_ct_args)
    reader_ct_args.extend(beta_ta_ct_args)

    # Runtime per core: input_addr, gamma_addr, beta_addr, scaler_bits, eps_bits, num_rows, start_row
    reader_rt_args = ttnn.RuntimeArgs()
    current_block = 0

    def _set_reader_rt_args(core_ranges, blocks_per_core):
        nonlocal current_block
        for core_range in core_ranges.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    start_row = current_block * 32
                    num_rows = blocks_per_core * 32
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        gamma.buffer_address() if gamma is not None else 0,
                        beta.buffer_address() if beta is not None else 0,
                        scaler_bits,
                        eps_bits,
                        num_rows,
                        start_row,
                    ]
                    current_block += blocks_per_core

    _set_reader_rt_args(core_group_1, blocks_per_core_g1)
    if has_group_2:
        _set_reader_rt_args(core_group_2, blocks_per_core_g2)

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # Compile-time: row_bytes, <output_ta>
    writer_ct_args = [row_bytes]
    writer_ct_args.extend(output_ta_ct_args)

    # Runtime per core: output_addr, num_rows, start_row
    writer_rt_args = ttnn.RuntimeArgs()
    current_block = 0

    def _set_writer_rt_args(core_ranges, blocks_per_core):
        nonlocal current_block
        for core_range in core_ranges.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    start_row = current_block * 32
                    num_rows = blocks_per_core * 32
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        num_rows,
                        start_row,
                    ]
                    current_block += blocks_per_core

    _set_writer_rt_args(core_group_1, blocks_per_core_g1)
    if has_group_2:
        _set_writer_rt_args(core_group_2, blocks_per_core_g2)

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel config ---
    compute_config_kwargs = {}
    use_fp32_dest = 0
    if compute_kernel_config is not None:
        if "math_fidelity" in compute_kernel_config:
            compute_config_kwargs["math_fidelity"] = compute_kernel_config["math_fidelity"]
        if "fp32_dest_acc_en" in compute_kernel_config:
            compute_config_kwargs["fp32_dest_acc_en"] = compute_kernel_config["fp32_dest_acc_en"]
            if compute_kernel_config["fp32_dest_acc_en"]:
                use_fp32_dest = 1
        if "math_approx_mode" in compute_kernel_config:
            compute_config_kwargs["math_approx_mode"] = compute_kernel_config["math_approx_mode"]
    compute_config_desc = ttnn.ComputeConfigDescriptor(**compute_config_kwargs)

    # --- Compute kernel(s) — one per core group ---
    kernels = [reader_kernel, writer_kernel]

    compute_ct_args_g1 = [Wt, blocks_per_core_g1, has_gamma, has_beta, use_fp32_dest]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
            core_ranges=core_group_1,
            compile_time_args=compute_ct_args_g1,
            runtime_args=[],
            config=compute_config_desc,
        )
    )

    if has_group_2:
        compute_ct_args_g2 = [Wt, blocks_per_core_g2, has_gamma, has_beta, use_fp32_dest]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
                core_ranges=core_group_2,
                compile_time_args=compute_ct_args_g2,
                runtime_args=[],
                config=compute_config_desc,
            )
        )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=cbs,
    )
