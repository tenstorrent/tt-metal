# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Configures circular buffers, kernels, and runtime args for the layer_norm_rm operation.

Work unit: 1 tile-row = 32 RM sticks spanning Wt tiles in the W dimension.
Work distribution: 1D linearized cores, each core handles ceil(nblocks / ncores) tile-rows.

CB layout (per design doc):
  c_0  (cb_in_rm)       - Wt pages  - RM sticks for tilize input
  c_1  (cb_tilized)     - Wt tiles  - Tilized input / Phase 8 output (reused)
  c_2  (cb_scaler)      - 1 tile    - Reduce scaler (1/W)
  c_3  (cb_eps)         - 1 tile    - Epsilon scalar tile
  c_4  (cb_gamma)       - Wt tiles  - Gamma affine tiles (persistent)
  c_5  (cb_beta)        - Wt tiles  - Beta affine tiles (persistent)
  c_16 (cb_out_rm)      - Wt pages  - Untilized RM output
  c_24 (cb_mean / temp) - 1 tile    - Row mean / var+eps / inv_std (multi-use)
  c_25 (cb_centered)    - Wt tiles  - x - mean (centered)
  c_26 (cb_centered_sq) - Wt tiles  - (x - mean)^2
  c_27 (cb_inv_std_tmp) - 1 tile    - var+eps intermediate
  c_28 (cb_normed)      - Wt tiles  - x_norm
  c_29 (cb_scaled)      - Wt tiles  - gamma * x_norm

Kernel paths: relative to tt-metal base directory (as expected by generic_op).
"""

import struct
import math
from pathlib import Path

import ttnn

# Kernel files: path relative to repository root (required by generic_op kernel_source)
# The kernel_source path must be relative to the tt-metal base directory
_OP_DIR = Path(__file__).parent
# Path relative to repo root -- computed at module load time
_REPO_ROOT = _OP_DIR
# Walk up from ttnn/ttnn/operations/layer_norm_rm to find repo root
# Structure: <repo_root>/ttnn/ttnn/operations/layer_norm_rm/
for _ in range(4):
    _REPO_ROOT = _REPO_ROOT.parent

KERNEL_DIR_REL = "ttnn/ttnn/operations/layer_norm_rm/kernels"

# CB indices per design doc
CB_IN_RM = 0  # RM sticks for tilize
CB_TILIZED = 1  # Tilized input / Phase 8 output (reused)
CB_SCALER = 2  # Reduce scaler (1/W)
CB_EPS = 3  # Epsilon scalar tile
CB_GAMMA = 4  # Gamma affine tiles
CB_BETA = 5  # Beta affine tiles
CB_OUT_RM = 16  # Untilized RM output
CB_MEAN = 24  # Row mean / variance / inv_std (multi-use temp)
CB_CENTERED = 25  # x - mean
CB_CENTERED_SQ = 26  # (x - mean)^2
CB_INV_STD_TMP = 27  # var + eps temp (before rsqrt)
CB_NORMED = 28  # x_norm
CB_SCALED = 29  # gamma * x_norm


def _float_to_uint32(value: float) -> int:
    """Bit-cast a float to uint32 (for passing float values as runtime args)."""
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-6,
) -> ttnn.ProgramDescriptor:
    """
    Build the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor:  Input bfloat16 RM tensor on device.
        output_tensor: Pre-allocated output bfloat16 RM tensor on device.
        gamma:         Optional bfloat16 RM gamma tensor, shape (1,1,1,W).
        beta:          Optional bfloat16 RM beta tensor, shape (1,1,1,W).
        epsilon:       Numerical stability constant.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op.
    """
    # ===== 1. SHAPE AND LAYOUT =====
    shape = input_tensor.shape
    rank = len(shape)

    # Extract last two dimensions (H, W) -- handle any rank >= 2
    W = shape[rank - 1]
    H = shape[rank - 2]

    # Total number of rows in the full tensor (product of all dims except last)
    # e.g. for (4, 2, 64, 128): outer = 4*2*2 = 16 blocks of 32 rows each
    total_rows = 1
    for i in range(rank - 1):
        total_rows *= shape[i]

    # Number of tile-rows (each tile-row = 32 RM rows = Wt tiles across W)
    Wt = W // 32  # tiles per row
    nblocks = total_rows // 32  # total tile-rows (each = 32 physical rows)

    # Tile size in bytes (bfloat16 32x32 tile = 2048 bytes)
    tile_size = ttnn.tile_size(input_tensor.dtype)
    # Element size in bytes: tile_size / (32 * 32)
    element_size_bytes = tile_size // (32 * 32)
    stick_size = W * element_size_bytes  # bytes per RM row

    # ===== 2. WORK DISTRIBUTION =====
    device = input_tensor.device()
    compute_grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        nblocks_per_core_group_1,
        nblocks_per_core_group_2,
    ) = ttnn.split_work_to_cores(compute_grid_size, nblocks)

    grid_width = compute_grid_size.x
    grid_height = compute_grid_size.y

    # Build a mapping from core -> num_blocks for this core
    # core_group_1 cores get nblocks_per_core_group_1 blocks
    # core_group_2 cores get nblocks_per_core_group_2 blocks
    def _core_in_range_set(core_range_set, x, y):
        """Check if (x, y) is in a CoreRangeSet using the contains() API."""
        return core_range_set.contains(ttnn.CoreCoord(x, y))

    # ===== 3. CB DESCRIPTORS =====
    # All CBs except c_0 and c_16 are tile-format.
    # c_0 and c_16 use RM sticks but same byte count as Wt tiles.

    # c_0: RM input sticks (Wt pages, each page = tile_size bytes = 32 sticks * W*2 bytes)
    cb_in_rm = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_IN_RM,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_1: Tilized input / Phase 8 output reuse (Wt tiles)
    cb_tilized = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_TILIZED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_2: Reduce scaler (1/W) -- single tile, persistent
    cb_scaler = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_3: Epsilon scalar tile -- single tile, persistent
    cb_eps = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EPS,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_4: Gamma tiles (Wt tiles, persistent) -- only allocated when gamma present
    cb_gamma = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_5: Beta tiles (Wt tiles, persistent) -- only allocated when beta present
    cb_beta = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_16: Untilized RM output (Wt pages, same byte count as Wt tiles)
    cb_out_rm = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT_RM,
                data_format=output_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_24: Row mean / var / inv_std temp (1 tile, multi-reuse)
    cb_mean = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_25: Centered values x - mean (Wt tiles)
    cb_centered = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_CENTERED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_26: Squared centered values (x - mean)^2 (Wt tiles)
    cb_centered_sq = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_CENTERED_SQ,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_27: var+eps intermediate (1 tile)
    cb_inv_std_tmp = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INV_STD_TMP,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_28: Normalized output x_norm (Wt tiles)
    cb_normed = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_NORMED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_29: Scaled output gamma * x_norm (Wt tiles)
    cb_scaled = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # ===== 4. COMMON VALUES =====
    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    has_gamma = 1 if gamma is not None else 0
    has_beta = 1 if beta is not None else 0

    # Epsilon and scaler: bit-cast floats to uint32
    eps_bits = _float_to_uint32(float(epsilon))
    scaler_bits = _float_to_uint32(1.0 / float(W))

    # ===== 5. KERNEL COMPILE-TIME ARGS =====

    # Reader compile-time args:
    #   [0] stick_size (bytes per RM row)
    #   [1+] TensorAccessorArgs(input_tensor)
    #   [N] has_gamma (0 or 1)
    #   [N+1] has_beta (0 or 1)
    #   [N+2+] TensorAccessorArgs(gamma) if has_gamma
    #   [M+] TensorAccessorArgs(beta) if has_beta
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Record index where has_gamma flag goes
    reader_ct_args.append(has_gamma)
    reader_ct_args.append(has_beta)
    if gamma is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    if beta is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    # Writer compile-time args:
    #   [0] stick_size
    #   [1] Wt
    #   [2+] TensorAccessorArgs(output_tensor)
    writer_ct_args = [stick_size, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute compile-time args:
    #   [0] Wt
    #   [1] has_gamma
    #   [2] has_beta
    compute_ct_args = [Wt, has_gamma, has_beta]

    # ===== 6. RUNTIME ARGS (per core) =====
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    # Track starting stick index (each tile-row = 32 sticks)
    start_stick = 0
    core_idx = 0

    for y in range(grid_height):
        for x in range(grid_width):
            in_group1 = _core_in_range_set(core_group_1, x, y)
            in_group2 = _core_in_range_set(core_group_2, x, y)

            if in_group1:
                num_blocks_this_core = nblocks_per_core_group_1
            elif in_group2:
                num_blocks_this_core = nblocks_per_core_group_2
            else:
                # Idle core: set empty args
                reader_rt_args[x][y] = []
                writer_rt_args[x][y] = []
                compute_rt_args[x][y] = []
                continue

            # Reader runtime args:
            #   [0] src_addr
            #   [1] num_blocks (tile-rows for this core)
            #   [2] start_stick_id
            #   [3] gamma_addr
            #   [4] beta_addr
            #   [5] eps_value (bit-cast uint32)
            #   [6] mean_scaler_value (bit-cast uint32)
            reader_rt_args[x][y] = [
                src_addr,
                num_blocks_this_core,
                start_stick * 32,  # start_stick * 32 rows per tile-row
                gamma_addr,
                beta_addr,
                eps_bits,
                scaler_bits,
            ]

            # Writer runtime args:
            #   [0] dst_addr
            #   [1] num_blocks (tile-rows for this core)
            #   [2] start_stick_id
            writer_rt_args[x][y] = [
                dst_addr,
                num_blocks_this_core,
                start_stick * 32,  # start output stick index
            ]

            # Compute runtime args:
            #   [0] num_blocks (tile-rows)
            compute_rt_args[x][y] = [
                num_blocks_this_core,
            ]

            start_stick += num_blocks_this_core
            core_idx += 1

    # ===== 7. KERNEL DESCRIPTORS =====

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR_REL}/layer_norm_rm_reader.cpp",
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR_REL}/layer_norm_rm_writer.cpp",
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR_REL}/layer_norm_rm_compute.cpp",
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ===== 8. ASSEMBLE PROGRAM DESCRIPTOR =====
    # Always include all CBs; the kernel writer controls which are actively used per stage
    all_cbs = [
        cb_in_rm,
        cb_tilized,
        cb_scaler,
        cb_eps,
        cb_gamma,
        cb_beta,
        cb_out_rm,
        cb_mean,
        cb_centered,
        cb_centered_sq,
        cb_inv_std_tmp,
        cb_normed,
        cb_scaled,
    ]

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
