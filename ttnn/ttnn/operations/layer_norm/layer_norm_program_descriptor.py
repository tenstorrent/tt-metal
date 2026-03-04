# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Program Descriptor

Defines circular buffers, kernel descriptors, and runtime args for
the three-pass layer norm operation:
  Pass 1: Compute mean via cross-tile accumulation + reduce_row
  Pass 2: Compute variance, add eps, rsqrt
  Pass 3: Normalize, optionally scale (gamma) and shift (beta)
"""

import struct
from pathlib import Path
import ttnn

# Kernel files are in the kernels/ subdirectory, relative to this file
_THIS_DIR = Path(__file__).parent
KERNEL_DIR = _THIS_DIR / "kernels"

# CB indices per design
CB_INPUT = 0  # c_0: input tile streaming
CB_SCALER = 1  # c_1: reduce scaler (1/W), filled once
CB_EPS = 2  # c_2: epsilon tile, filled once
CB_GAMMA = 3  # c_3: gamma tile (pass 3 only)
CB_BETA = 4  # c_4: beta tile (pass 3 only)
CB_OUTPUT = 16  # c_16: output tile streaming
CB_MEAN = 24  # c_24: mean per row (persists across passes)
CB_ACCUM = 25  # c_25: cross-tile accumulator
CB_VAR = 26  # c_26: variance + eps (rsqrt result)
CB_TMP = 27  # c_27: scratch tile


def _float_to_bfloat16_bits(value: float) -> int:
    """Convert a Python float to its bfloat16 bit representation (uint16)."""
    # Pack as float32, take upper 2 bytes (= bfloat16)
    packed = struct.pack(">f", value)
    bf16_bits = (packed[0] << 8) | packed[1]
    return bf16_bits


def _pack_bfloat16_pair(value: float) -> int:
    """
    Pack a float into the (bf16 << 16 | bf16) uint32 format used by
    the prepare_reduce_scaler helper for filling a full 32x32 tile.
    """
    bf16 = _float_to_bfloat16_bits(value)
    return (bf16 << 16) | bf16


def _make_bf16_tile_cb(cb_index: int, core_ranges: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    """Create a single-page bfloat16 tile CB (2048 bytes)."""
    tile_size = 2048  # bf16, 32x32 tile
    return ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Build the ProgramDescriptor for layer norm.

    Args:
        input_tensor:  Input tensor (BFLOAT16, TILE_LAYOUT, on device).
        output_tensor: Pre-allocated output tensor (same spec as input).
        gamma:         Optional scale tensor [1,1,1,W].
        beta:          Optional bias tensor  [1,1,1,W].
        eps:           Variance stability constant.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op().
    """
    # ======================================================================
    # 1. Tensor metadata
    # ======================================================================
    shape = input_tensor.shape  # ttnn.Shape, must be indexed element by element
    rank = len(shape)

    # Treat the tensor as [N, W] internally.
    # For a 4D tensor [B, C, H, W] the "rows" we normalize = B*C*H / TILE_H rows of tiles.
    # W is always shape[-1], H (tile rows) comes from the product of remaining dims / 32.
    W = shape[rank - 1]  # width in elements (multiple of 32)
    Wt = W // 32  # width in tiles

    # Total number of rows in terms of 32-element tile rows.
    # For 4D [b, c, h, w]: N_rows_of_tiles = b * c * (h // 32)
    # For 2D [h, w]: N_rows_of_tiles = h // 32
    total_elements = 1
    for i in range(rank - 1):
        total_elements *= shape[i]
    # total_elements is the product of all dims except W
    # Divide by 32 to get number of tile-rows
    N_tile_rows = total_elements // 32  # each tile row is 32 elements tall

    tile_size = 2048  # bytes per bf16 32x32 tile

    # ======================================================================
    # 2. Work distribution
    # ======================================================================
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    # split_work_to_cores takes a CoreRangeSet and total_work_units (tile rows here)
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(compute_grid, N_tile_rows)

    # ======================================================================
    # 3. Circular buffer descriptors (all single-buffered, 1 page, bf16)
    # ======================================================================
    cb_input_desc = _make_bf16_tile_cb(CB_INPUT, all_cores)
    cb_scaler_desc = _make_bf16_tile_cb(CB_SCALER, all_cores)
    cb_eps_desc = _make_bf16_tile_cb(CB_EPS, all_cores)
    cb_gamma_desc = _make_bf16_tile_cb(CB_GAMMA, all_cores)
    cb_beta_desc = _make_bf16_tile_cb(CB_BETA, all_cores)
    cb_output_desc = _make_bf16_tile_cb(CB_OUTPUT, all_cores)
    cb_mean_desc = _make_bf16_tile_cb(CB_MEAN, all_cores)
    cb_accum_desc = _make_bf16_tile_cb(CB_ACCUM, all_cores)
    cb_var_desc = _make_bf16_tile_cb(CB_VAR, all_cores)
    cb_tmp_desc = _make_bf16_tile_cb(CB_TMP, all_cores)

    cbs = [
        cb_input_desc,
        cb_scaler_desc,
        cb_eps_desc,
        cb_gamma_desc,
        cb_beta_desc,
        cb_output_desc,
        cb_mean_desc,
        cb_accum_desc,
        cb_var_desc,
        cb_tmp_desc,
    ]

    # ======================================================================
    # 4. Addresses and packed constants
    # ======================================================================
    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # eps packed as (bf16 << 16 | bf16) uint32
    eps_bits = _pack_bfloat16_pair(eps)

    # Scaler value: 1/W (encodes 1/W so reduce_row produces mean directly)
    scaler_bits = _pack_bfloat16_pair(1.0 / W)

    # ======================================================================
    # 5. Reader kernel descriptor
    # ======================================================================
    reader_ct_args = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if gamma is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    if beta is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()  # compute has no runtime args

    tile_offset = 0

    def _set_rt_args_for_group(core_group, rows_per_core):
        nonlocal tile_offset
        for core_range in core_group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_addr,
                        rows_per_core,
                        Wt,
                        tile_offset,
                        gamma_addr,
                        beta_addr,
                        eps_bits,
                        scaler_bits,
                    ]
                    writer_rt_args[x][y] = [
                        output_addr,
                        rows_per_core,
                        Wt,
                        tile_offset,
                    ]
                    compute_rt_args[x][y] = []
                    tile_offset += rows_per_core * Wt

    _set_rt_args_for_group(core_group_1, rows_per_core_g1)
    if len(core_group_2.ranges()) > 0:
        _set_rt_args_for_group(core_group_2, rows_per_core_g2)

    # ======================================================================
    # 6. Compute compile-time args (per core group, since num_rows differs)
    # ======================================================================
    compute_ct_args_g1 = [
        rows_per_core_g1,  # index 0: num_rows_per_core
        Wt,  # index 1: width in tiles
        has_gamma,  # index 2: 1 if gamma present
        has_beta,  # index 3: 1 if beta present
    ]

    has_group_2 = len(core_group_2.ranges()) > 0 and rows_per_core_g2 > 0

    if has_group_2:
        compute_ct_args_g2 = [
            rows_per_core_g2,  # index 0: num_rows_per_core (fewer for group 2)
            Wt,
            has_gamma,
            has_beta,
        ]

    writer_ct_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ======================================================================
    # 7. Kernel descriptors
    # ======================================================================
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_layer_norm.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_layer_norm.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )

    compute_kernel_g1 = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_layer_norm.cpp"),
        core_ranges=core_group_1,
        compile_time_args=compute_ct_args_g1,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    kernels = [reader_kernel, writer_kernel, compute_kernel_g1]

    if has_group_2:
        compute_kernel_g2 = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "compute_layer_norm.cpp"),
            core_ranges=core_group_2,
            compile_time_args=compute_ct_args_g2,
            runtime_args=compute_rt_args,
            config=compute_config,
        )
        kernels.append(compute_kernel_g2)

    # ======================================================================
    # 8. Assemble and return
    # ======================================================================
    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=cbs,
    )
