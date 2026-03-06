# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.

Architecture overview:
- Work unit: one tile-row (32 RM sticks of width W = Wt tiles)
- Total work: total_tile_rows = (batch dims product) * H // 32
- Distribution: linearized 1D grid split across cores

CB layout:
  c_0  (cb_in)          - RM input sticks (Wt tile-sized pages), Block lifetime
  c_1  (cb_gamma)       - Gamma row tiles (Wt), Program lifetime constant
  c_2  (cb_beta)        - Beta row tiles (Wt), Program lifetime constant
  c_8  (cb_reduce_scaler) - 1/W reduce scaler (1 tile), Program lifetime constant
  c_9  (cb_eps)         - Epsilon scalar tile (1 tile), Program lifetime constant
  c_16 (cb_out)         - RM output sticks (Wt tile-sized pages), Block lifetime
  c_24 (cb_tilized)     - Tilized input (Wt tiles), Row lifetime
  c_25 (cb_mean)        - Mean tile (1 tile), Row lifetime
  c_26 (cb_centered)    - x - mean (Wt tiles), Row lifetime
  c_27 (cb_squared)     - (x-mean)^2 (Wt tiles), Row lifetime
  c_28 (cb_var_eps)     - var + eps intermediate (1 tile), Row lifetime
  c_29 (cb_inv_std)     - rsqrt(var+eps) (1 tile), Row lifetime
  c_30 (cb_normed)      - centered * inv_std (Wt tiles), Row lifetime
  c_31 (cb_affine_out)  - gamma * normed (Wt tiles), Row lifetime (ping-pong source for +beta)
"""

import struct
from pathlib import Path
import ttnn

# Kernel files are relative to tt-metal base folder
_OP_DIR = Path(__file__).parent
# Path relative to tt-metal root for kernel source references
_OP_KERNEL_PATH = "ttnn/ttnn/operations/layer_norm_rm/kernels"

# CB indices
CB_IN = 0  # RM input sticks (tile-sized pages)
CB_GAMMA = 1  # Gamma RM stick (for tilize input)
CB_BETA = 2  # Beta RM stick (for tilize input)
CB_GAMMA_TILED = 3  # Gamma tiles after tilize (program-lifetime)
CB_BETA_TILED = 4  # Beta tiles after tilize (program-lifetime)
CB_REDUCE_SCALER = 8  # 1/W reduce scaler (constant, program-lifetime)
CB_EPS = 9  # Epsilon scalar tile (constant, program-lifetime)
CB_OUT = 16  # RM output sticks (tile-sized pages)
CB_TILIZED = 24  # Tilized input
CB_MEAN = 25  # Mean result from reduce
CB_CENTERED = 26  # x - mean
CB_SQUARED = 27  # (x - mean)^2
CB_VAR_EPS = 28  # var (intermediate, reused for var+eps)
CB_INV_STD = 29  # rsqrt(var + eps)
CB_NORMED = 30  # centered * inv_std (also final output ping-pong source)
CB_AFFINE_OUT = 31  # gamma * normed (before +beta)


def _float_to_packed_bf16_u32(val: float) -> int:
    """
    Pack a float as a bfloat16 scalar in the upper 16 bits of a uint32.

    This is the format expected by generate_reduce_scaler / prepare_reduce_scaler
    when called from the Python side (runtime arg). The scaler is stored as:
        (bf16_bits << 16) | bf16_bits
    """
    # Convert to float32 bytes, take upper 2 bytes (bf16 truncation)
    f32_bytes = struct.pack(">f", val)
    bf16_bits = (f32_bytes[0] << 8) | f32_bytes[1]
    return (bf16_bits << 16) | bf16_bits


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
        input_tensor:  Input tensor (ROW_MAJOR, bfloat16, on device)
        output_tensor: Pre-allocated output tensor (ROW_MAJOR, bfloat16, on device)
        gamma:         Optional gamma tensor (ROW_MAJOR, bfloat16, on device)
        beta:          Optional beta tensor (ROW_MAJOR, bfloat16, on device)
        epsilon:       Variance stabilization constant

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    ndim = len(shape)
    W = shape[ndim - 1]  # Last dim (width in elements)
    H = shape[ndim - 2]  # Second-to-last dim (height in elements)

    # Wt = number of 32-wide tiles per row
    Wt = W // 32
    # Ht = number of tile-rows per 2D slice
    Ht = H // 32

    # Total outer batch dimensions (product of all dims except last two)
    batch_size = 1
    for i in range(ndim - 2):
        batch_size *= shape[i]

    # Total tile-rows = batch_size * Ht
    total_tile_rows = batch_size * Ht

    # Stick size in bytes: one RM row of W bfloat16 elements
    # For ROW_MAJOR layout, buffer_page_size() == stick size (bytes per row)
    stick_size = input_tensor.buffer_page_size()

    # Tile size in bytes for bfloat16 32x32 tile (used for CB page_size)
    # For RM tensors, page_size for CB must still be tile-sized so tilize works
    tile_size = ttnn.tile_size(input_tensor.dtype)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tile_rows_per_core_group_1,
        tile_rows_per_core_group_2,
    ) = ttnn.split_work_to_cores(compute_grid_size, total_tile_rows)

    # Build list of active cores with per-core tile_row counts
    active_cores = []
    for core_range in core_group_1.ranges():
        start = core_range.start
        end = core_range.end
        for y in range(start.y, end.y + 1):
            for x in range(start.x, end.x + 1):
                active_cores.append((x, y, tile_rows_per_core_group_1))
    for core_range in core_group_2.ranges():
        start = core_range.start
        end = core_range.end
        for y in range(start.y, end.y + 1):
            for x in range(start.x, end.x + 1):
                active_cores.append((x, y, tile_rows_per_core_group_2))

    # Build set of all core coordinates in grid for idle-core zeroing
    grid_w = compute_grid_size.x
    grid_h = compute_grid_size.y
    active_core_set = {(cx, cy) for cx, cy, _ in active_cores}

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # For ROW_MAJOR tensors doing tilize/untilize:
    # - cb_in and cb_out use tile_size as page_size (tilize reads/writes tile-sized pages)
    # - All intermediate CBs use tile_size
    # - gamma/beta CBs: Wt tiles of tile_size each

    # cb_in: Wt tile-sized pages (32 RM sticks grouped as Wt tile pages)
    cb_in_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_IN,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_gamma: Wt tile-sized pages (program-lifetime constant)
    cb_gamma_descriptor = ttnn.CBDescriptor(
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

    # cb_beta: Wt tile-sized pages (program-lifetime constant)
    cb_beta_descriptor = ttnn.CBDescriptor(
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

    # cb_gamma_tiled: Wt tiles (tilized gamma, program-lifetime)
    cb_gamma_tiled_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA_TILED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_beta_tiled: Wt tiles (tilized beta, program-lifetime)
    cb_beta_tiled_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA_TILED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_reduce_scaler: 1 tile (program-lifetime constant)
    cb_reduce_scaler_descriptor = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_REDUCE_SCALER,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_eps: 1 tile (program-lifetime constant)
    cb_eps_descriptor = ttnn.CBDescriptor(
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

    # cb_out: Wt tile-sized pages (for untilize output)
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT,
                data_format=output_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_tilized: Wt tiles (intermediate)
    cb_tilized_descriptor = ttnn.CBDescriptor(
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

    # cb_mean: 1 tile (mean result from row reduce)
    cb_mean_descriptor = ttnn.CBDescriptor(
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

    # cb_centered: Wt tiles (x - mean)
    cb_centered_descriptor = ttnn.CBDescriptor(
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

    # cb_squared: Wt tiles ((x-mean)^2)
    cb_squared_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SQUARED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_var_eps: 1 tile (variance, then var+eps)
    cb_var_eps_descriptor = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_VAR_EPS,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_inv_std: 1 tile (rsqrt(var+eps))
    cb_inv_std_descriptor = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INV_STD,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # cb_normed: Wt tiles (centered * inv_std, reused for +beta result)
    cb_normed_descriptor = ttnn.CBDescriptor(
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

    # cb_affine_out: Wt tiles (gamma * normed, before +beta)
    cb_affine_out_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_AFFINE_OUT,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    all_cbs = [
        cb_in_descriptor,
        cb_gamma_descriptor,
        cb_beta_descriptor,
        cb_gamma_tiled_descriptor,
        cb_beta_tiled_descriptor,
        cb_reduce_scaler_descriptor,
        cb_eps_descriptor,
        cb_out_descriptor,
        cb_tilized_descriptor,
        cb_mean_descriptor,
        cb_centered_descriptor,
        cb_squared_descriptor,
        cb_var_eps_descriptor,
        cb_inv_std_descriptor,
        cb_normed_descriptor,
        cb_affine_out_descriptor,
    ]

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    # Compile-time args:
    #   [0] stick_size       - Width of one RM stick in bytes
    #   [1] Wt               - Tiles per row
    #   [2] has_gamma        - 1 if gamma tensor provided, else 0
    #   [3] has_beta         - 1 if beta tensor provided, else 0
    #   [4+] input TensorAccessor compile-time args
    #   [4+N] gamma TensorAccessor compile-time args (if gamma)
    #   [4+N+M] beta TensorAccessor compile-time args (if beta)
    reader_ct_args = [
        stick_size,
        Wt,
        1 if gamma is not None else 0,
        1 if beta is not None else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if gamma is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    if beta is not None:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    # Pack float values as uint32 (IEEE 754 bits) for passing to kernel
    eps_packed = struct.unpack("I", struct.pack("f", epsilon))[0]
    scaler_val = 1.0 / float(W)
    scaler_packed = struct.unpack("I", struct.pack("f", scaler_val))[0]

    reader_rt_args = ttnn.RuntimeArgs()

    # Set reader runtime args per core
    start_tile_row = 0
    for cx, cy, n_tile_rows in active_cores:
        # start_stick_id = start_tile_row * 32  (each tile-row covers 32 sticks)
        start_stick_id = start_tile_row * 32
        args = [
            input_tensor.buffer_address(),  # src_addr
            n_tile_rows,  # N (tile-rows for this core)
            start_stick_id,  # start_stick_id
            scaler_packed,  # scaler value for reduce (1/W)
            eps_packed,  # epsilon value
            gamma.buffer_address() if gamma is not None else 0,  # gamma_addr
            beta.buffer_address() if beta is not None else 0,  # beta_addr
        ]
        reader_rt_args[cx][cy] = args
        start_tile_row += n_tile_rows

    # Set empty args for idle cores
    for y in range(grid_h):
        for x in range(grid_w):
            if (x, y) not in active_core_set:
                reader_rt_args[x][y] = []

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_OP_KERNEL_PATH}/layer_norm_rm_reader.cpp",
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # Compile-time args:
    #   [0] cb_id_out              - Output CB index (CB_OUT = 16)
    #   [1] output_stick_size      - Bytes per output stick
    #   [2] tile_height            - 32 (rows per tile)
    #   [3] num_tiles_per_row      - Wt
    #   [4+] output TensorAccessor compile-time args
    writer_ct_args = [
        CB_OUT,
        stick_size,
        32,  # tile_height
        Wt,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()

    start_tile_row = 0
    for cx, cy, n_tile_rows in active_cores:
        start_stick_id = start_tile_row * 32
        args = [
            output_tensor.buffer_address(),  # dst_addr
            n_tile_rows,  # N (tile-rows for this core)
            start_stick_id,  # start_stick_id
        ]
        writer_rt_args[cx][cy] = args
        start_tile_row += n_tile_rows

    # Set empty args for idle cores
    for y in range(grid_h):
        for x in range(grid_w):
            if (x, y) not in active_core_set:
                writer_rt_args[x][y] = []

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_OP_KERNEL_PATH}/layer_norm_rm_writer.cpp",
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Compile-time args:
    #   [0] Wt            - Tiles per tile-row (width)
    #   [1] has_gamma     - 1 if gamma, else 0
    #   [2] has_beta      - 1 if beta, else 0
    compute_ct_args = [
        Wt,
        1 if gamma is not None else 0,
        1 if beta is not None else 0,
    ]

    compute_rt_args = ttnn.RuntimeArgs()

    # Compute kernel gets N (tile-rows per core) as a runtime arg
    start_tile_row = 0
    for cx, cy, n_tile_rows in active_cores:
        compute_rt_args[cx][cy] = [n_tile_rows]
        start_tile_row += n_tile_rows

    # Set empty args for idle cores
    for y in range(grid_h):
        for x in range(grid_w):
            if (x, y) not in active_core_set:
                compute_rt_args[x][y] = []

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_OP_KERNEL_PATH}/layer_norm_rm_compute.cpp",
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
