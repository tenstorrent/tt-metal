# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args
for single-core layer normalization over the last dimension of a row-major tensor.

Optimized CB Layout (CB reuse for non-overlapping lifetimes):
  CB  0 (cb_input_rm)      : RM input sticks,        page_size=stick_size,  pages=32
  CB  1 (cb_gamma_rm)      : RM gamma stick,          page_size=stick_size,  pages=1
  CB  2 (cb_beta_rm)       : RM beta stick,           page_size=stick_size,  pages=1
  CB  8 (cb_reduce_scaler) : reduce scaler tile,      page_size=tile_size,   pages=1
  CB  9 (cb_eps_scaler)    : epsilon scaler tile,     page_size=tile_size,   pages=1
  CB 16 (cb_output_rm)     : RM output (post-untilize), page_size=tile_size, pages=Wt
  CB 24 (cb_tilized/norm)  : tilized input + normalized (multi-use), page_size=tile_size, pages=Wt
  CB 25 (cb_mean/var_sum)  : mean + variance sum (multi-use), page_size=tile_size, pages=1
  CB 26 (cb_centered)      : centered values,         page_size=tile_size,   pages=Wt
  CB 28 (cb_gamma_tilized) : tilized gamma,           page_size=tile_size,   pages=Wt
  CB 29 (cb_beta_tilized)  : tilized beta,            page_size=tile_size,   pages=Wt
  CB 30 (cb_rstd)          : reciprocal std dev,      page_size=tile_size,   pages=1
"""

import struct
from pathlib import Path
import ttnn

# Kernel files are relative to the tt-metal repo root (kernel_source is relative to that)
_OP_DIR = Path(__file__).parent
_KERNEL_REPO_PATH = "ttnn/ttnn/operations/layer_norm_rm/kernels"

# CB indices
CB_INPUT_RM = 0
CB_GAMMA_RM = 1
CB_BETA_RM = 2
CB_REDUCE_SCALER = 8
CB_EPS_SCALER = 9
CB_OUTPUT_RM = 16
CB_TILIZED = 24  # Multi-use: tilized input (phase 1-3) + normalized (phases 4-10)
CB_MEAN = 25  # Multi-use: mean (phase 2-3) + var_sum (phase 5-6)
CB_CENTERED = 26
CB_GAMMA_TILIZED = 28
CB_BETA_TILIZED = 29
CB_RSTD = 30

# Tile size for bfloat16: 32x32 * 2 bytes = 2048
TILE_SIZE_BF16 = 2048


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor,
    beta_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    epsilon: float = 1e-6,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the layer_norm_rm operation.

    Args:
        input_tensor:  Input tensor (bfloat16, ROW_MAJOR, DRAM, shape [N,C,H,W])
        gamma_tensor:  Gamma tensor (bfloat16, ROW_MAJOR, DRAM, shape [1,1,1,W])
        beta_tensor:   Beta tensor  (bfloat16, ROW_MAJOR, DRAM, shape [1,1,1,W])
        output_tensor: Pre-allocated output tensor (same shape as input)
        epsilon:       Variance stability constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. TENSOR METADATA ==========
    shape = input_tensor.shape
    # shape dimensions
    rank = len(shape)
    W = shape[-1]
    H = shape[-2]
    # Compute N*C (all batch dimensions combined)
    NC = 1
    for i in range(rank - 2):
        NC *= shape[i]

    Wt = W // 32  # tiles per row
    Ht = NC * H // 32  # total tile rows (N*C*H / 32)

    stick_size = W * 2  # bytes: W elements * 2 bytes (bfloat16)
    tile_size = TILE_SIZE_BF16

    num_sticks_total = NC * H  # N*C*H total input sticks
    num_gamma_sticks = 1  # Asymmetric tilize: 1 stick, ROW broadcast

    # Convert epsilon to uint32 bit representation
    epsilon_bits = struct.unpack("I", struct.pack("f", epsilon))[0]

    # ========== 2. CORE GRID ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    cbs = []

    # CB 0: cb_input_rm - RM input sticks.
    # 32 sticks per tile-row, each of stick_size bytes.
    cbs.append(_make_cb(CB_INPUT_RM, ttnn.bfloat16, stick_size, 32, core_grid))

    # CB 1: cb_gamma_rm - RM gamma stick (1 stick for asymmetric tilize)
    cbs.append(_make_cb(CB_GAMMA_RM, ttnn.bfloat16, stick_size, 1, core_grid))

    # CB 2: cb_beta_rm - RM beta stick (1 stick for asymmetric tilize)
    cbs.append(_make_cb(CB_BETA_RM, ttnn.bfloat16, stick_size, 1, core_grid))

    # CB 8: cb_reduce_scaler - reduce scaler tile (tile_size, 1 page)
    cbs.append(_make_cb(CB_REDUCE_SCALER, ttnn.bfloat16, tile_size, 1, core_grid))

    # CB 9: cb_eps_scaler - epsilon scaler tile (tile_size, 1 page)
    cbs.append(_make_cb(CB_EPS_SCALER, ttnn.bfloat16, tile_size, 1, core_grid))

    # CB 16: cb_output_rm - output after untilize (tile_size pages, Wt pages)
    cbs.append(_make_cb(CB_OUTPUT_RM, ttnn.bfloat16, tile_size, Wt, core_grid))

    # CB 24: multi-use (tilized input + normalized) - Wt tiles
    cbs.append(_make_cb(CB_TILIZED, ttnn.bfloat16, tile_size, Wt, core_grid))

    # CB 25: multi-use (mean + var_sum) - 1 tile
    cbs.append(_make_cb(CB_MEAN, ttnn.bfloat16, tile_size, 1, core_grid))

    # CB 26: cb_centered - centered values (Wt tiles)
    cbs.append(_make_cb(CB_CENTERED, ttnn.bfloat16, tile_size, Wt, core_grid))

    # CB 28: cb_gamma_tilized - tilized gamma (Wt tiles)
    cbs.append(_make_cb(CB_GAMMA_TILIZED, ttnn.bfloat16, tile_size, Wt, core_grid))

    # CB 29: cb_beta_tilized - tilized beta (Wt tiles)
    cbs.append(_make_cb(CB_BETA_TILIZED, ttnn.bfloat16, tile_size, Wt, core_grid))

    # CB 30: cb_rstd - reciprocal std dev (1 tile)
    cbs.append(_make_cb(CB_RSTD, ttnn.bfloat16, tile_size, 1, core_grid))

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    reader_ct_args = [
        stick_size,  # index 0: stick_size (W*2 bytes)
        num_sticks_total,  # index 1: num_sticks_total (N*C*H)
        num_gamma_sticks,  # index 2: num_gamma_sticks (1)
        Wt,  # index 3: Wt (W/32)
        epsilon_bits,  # index 4: epsilon as uint32 bits
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),  # index 0: input_addr
        gamma_tensor.buffer_address(),  # index 1: gamma_addr
        beta_tensor.buffer_address(),  # index 2: beta_addr
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_REPO_PATH}/reader_layer_norm_rm.cpp",
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [
        Ht,  # index 0: Ht (total tile rows)
        Wt,  # index 1: Wt (tiles per row)
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_REPO_PATH}/compute_layer_norm_rm.cpp",
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
        stick_size,  # index 0: stick_size
        num_sticks_total,  # index 1: num_sticks_total
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),  # index 0: output_addr
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNEL_REPO_PATH}/writer_layer_norm_rm.cpp",
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )


def _make_cb(cb_id: int, dtype, page_size: int, num_pages: int, core_grid) -> ttnn.CBDescriptor:
    """Helper to create a CBDescriptor with a single format descriptor."""
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_id,
                data_format=dtype,
                page_size=page_size,
            )
        ],
    )
