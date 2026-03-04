# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines circular buffers, kernel descriptors and runtime args for the
single-core layer normalization over row-major interleaved tensors.

CB layout (indices match kernel source comments):
  c_0  (cb_input_rm)     - Input RM sticks (Wt pages, tile-sized each)
  c_1  (cb_gamma_rm)     - Gamma RM sticks (Wt pages)
  c_2  (cb_beta_rm)      - Beta  RM sticks (Wt pages)
  c_8  (cb_scaler)       - Reduce scaler 1/W, 1 tile, never popped
  c_9  (cb_eps)          - Epsilon scaler tile, 1 tile, never popped
  c_16 (cb_out_rm)       - Output RM sticks (Wt pages)
  c_24 (cb_input_tiled)  - Tilized input (Wt tiles)
  c_25 (cb_mean)         - Mean tile (1 tile)
  c_26 (cb_centered)     - x - mean (Wt tiles)
  c_27 (cb_var_sq)       - centered^2 (Wt tiles)
  c_28 (cb_inv_std)      - inv_std tile (1 tile)
  c_29 (cb_gamma_tiled)  - Tilized gamma (Wt tiles, program lifetime)
  c_30 (cb_beta_tiled)   - Tilized beta  (Wt tiles, program lifetime)
  c_31 (cb_normed)       - x_centered * inv_std (Wt tiles)
"""

import struct
from pathlib import Path
import ttnn

# Kernel sources are relative paths from the tt-metal base directory
_OP_DIR = Path(__file__).parent
KERNEL_DIR = _OP_DIR / "kernels"

# Relative path prefix (from tt-metal base) for kernel_source strings
# generic_op resolves kernel_source relative to the tt-metal repo root.
# We compute an absolute path and pass it; the framework accepts absolute paths.
_READER_KERNEL = str(KERNEL_DIR / "layer_norm_rm_reader.cpp")
_COMPUTE_KERNEL = str(KERNEL_DIR / "layer_norm_rm_compute.cpp")
_WRITER_KERNEL = str(KERNEL_DIR / "layer_norm_rm_writer.cpp")

# CB index constants
CB_INPUT_RM = 0
CB_GAMMA_RM = 1
CB_BETA_RM = 2
CB_SCALER = 8
CB_EPS = 9
CB_OUT_RM = 16
CB_INPUT_TILED = 24
CB_MEAN = 25
CB_CENTERED = 26
CB_VAR_SQ = 27
CB_INV_STD = 28
CB_GAMMA_TILED = 29
CB_BETA_TILED = 30
CB_NORMED = 31


def _float_to_uint32_bits(value: float) -> int:
    """Reinterpret float32 as uint32 bits for kernel __builtin_bit_cast."""
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor,
    beta_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    epsilon: float = 1e-5,
    bisect_phase: int = 99,
) -> ttnn.ProgramDescriptor:
    """
    Build the ProgramDescriptor for layer_norm_rm.

    Single-core implementation. The core processes all tile-rows sequentially.
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    rank = len(input_tensor.shape)
    W = input_tensor.shape[rank - 1]
    H = input_tensor.shape[rank - 2]

    # Number of outer batch dimensions
    N_outer = 1
    for i in range(rank - 2):
        N_outer *= input_tensor.shape[i]

    Wt = W // 32  # tiles per row
    Ht = H // 32  # tile-rows per H
    num_rows = N_outer * Ht  # total tile-rows (each is 32 physical rows tall)

    # Stick size in bytes (one row of W bfloat16 elements)
    # element_size is not directly exposed; derive from tile_size / (32*32)
    tile_size = ttnn.tile_size(input_tensor.dtype)  # e.g. 2048 bytes for bf16
    element_size = tile_size // (32 * 32)
    stick_size = W * element_size  # W * 2 for bf16

    # RM page size equals stick_size rounded up to DRAM alignment
    # For CBs, use tile_size as the page size (symmetric tilize mode: one
    # RM "page" covers the data for one tile column, so 32 sticks of width W/Wt).
    # The RM input CBs store Wt "pages" where each page = stick_size
    # (the tilize helper reads the full row and produces Wt tiles).
    # We allocate Wt pages of tile_size for RM CBs to match the tilize helper
    # requirement (symmetric tilize: each CB page = one tile worth of RM data).
    rm_cb_page_size = tile_size  # One tile-sized slot per column tile

    # ========== 2. SINGLE-CORE GRID ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========

    # --- Flowing CBs (block lifetime, Wt pages each) ---
    cb_input_rm = ttnn.CBDescriptor(
        total_size=Wt * rm_cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_RM,
                data_format=input_tensor.dtype,
                page_size=rm_cb_page_size,
            )
        ],
    )

    cb_gamma_rm = ttnn.CBDescriptor(
        total_size=Wt * rm_cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA_RM,
                data_format=gamma_tensor.dtype,
                page_size=rm_cb_page_size,
            )
        ],
    )

    cb_beta_rm = ttnn.CBDescriptor(
        total_size=Wt * rm_cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA_RM,
                data_format=beta_tensor.dtype,
                page_size=rm_cb_page_size,
            )
        ],
    )

    # --- Scaler CBs (1 tile each, never popped) ---
    cb_scaler = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SCALER,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_eps = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_EPS,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    # --- Output RM CB (block lifetime, Wt pages) ---
    cb_out_rm = ttnn.CBDescriptor(
        total_size=Wt * rm_cb_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT_RM,
                data_format=output_tensor.dtype,
                page_size=rm_cb_page_size,
            )
        ],
    )

    # --- Intermediate tiled CBs ---
    cb_input_tiled = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_TILED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_mean = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_MEAN,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_centered = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_CENTERED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_var_sq = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_VAR_SQ,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_inv_std = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INV_STD,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_gamma_tiled = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_GAMMA_TILED,
                data_format=gamma_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_beta_tiled = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BETA_TILED,
                data_format=beta_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    cb_normed = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_NORMED,
                data_format=input_tensor.dtype,
                page_size=tile_size,
            )
        ],
    )

    all_cbs = [
        cb_input_rm,
        cb_gamma_rm,
        cb_beta_rm,
        cb_scaler,
        cb_eps,
        cb_out_rm,
        cb_input_tiled,
        cb_mean,
        cb_centered,
        cb_var_sq,
        cb_inv_std,
        cb_gamma_tiled,
        cb_beta_tiled,
        cb_normed,
    ]

    # ========== 4. KERNEL DESCRIPTORS ==========

    # --- Reader kernel ---
    # Compile-time args:
    #   [0]      stick_size            - bytes per RM stick (W * 2)
    #   [1..]    TensorAccessorArgs(input)
    #   [N+0]    gamma_stick_size      - same as stick_size
    #   [N+1..]  TensorAccessorArgs(gamma)
    #   [M+0]    beta_stick_size       - same as stick_size
    #   [M+1..]  TensorAccessorArgs(beta)
    reader_ct_args = [stick_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.append(stick_size)  # gamma_stick_size
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args())
    reader_ct_args.append(stick_size)  # beta_stick_size
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args())

    # Runtime args (single core):
    #   [0] src_addr        - input buffer address
    #   [1] gamma_addr      - gamma buffer address
    #   [2] beta_addr       - beta buffer address
    #   [3] num_rows        - total tile-rows to process
    #   [4] Wt              - tiles per row
    #   [5] start_stick_id  - first stick id for this core (0 for single core)
    #   [6] scaler_value    - 1/W as packed bf16 uint32
    #   [7] eps_value       - epsilon as packed bf16 uint32
    scaler_value = _float_to_uint32_bits(1.0 / W)
    eps_value = _float_to_uint32_bits(epsilon)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_tensor.buffer_address(),
        beta_tensor.buffer_address(),
        num_rows,
        Wt,
        0,  # start_stick_id
        scaler_value,
        eps_value,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # Compile-time args:
    #   [0]    stick_size              - bytes per output RM stick
    #   [1..]  TensorAccessorArgs(output)
    writer_ct_args = [stick_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Runtime args:
    #   [0] dst_addr        - output buffer address
    #   [1] num_rows        - total tile-rows to process
    #   [2] Wt              - tiles per row
    #   [3] start_stick_id  - first output stick id (0 for single core)
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_rows,
        Wt,
        0,  # start_stick_id
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Compile-time args:
    #   [0] num_rows   - total tile-rows to process
    #   [1] Wt         - tiles per row
    compute_ct_args = [num_rows, Wt]

    compute_defines = []
    if bisect_phase != 99:
        compute_defines.append(("BISECT_PHASE", str(bisect_phase)))

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        defines=compute_defines,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        ),
    )

    # ========== 5. ASSEMBLE PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
