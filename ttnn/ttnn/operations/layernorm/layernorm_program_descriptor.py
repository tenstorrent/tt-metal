# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Program Descriptor

Defines the ProgramDescriptor: 11 circular buffers, 3 kernels (reader, compute, writer),
and per-core runtime args for tile-row based work distribution.

CB layout:
  c_0  (cb_in)             - RM input sticks (Wt pages)
  c_1  (cb_tilized)        - Tilized input (Wt tiles)
  c_2  (cb_reduce_scaler)  - Reduce scaler 1/W (1 tile)
  c_3  (cb_mean)           - Row mean (1 tile)
  c_4  (cb_centered)       - x - mean (Wt tiles)
  c_5  (cb_var)            - Row variance (1 tile)
  c_6  (cb_gamma)          - Gamma tiles (Wt tiles)
  c_7  (cb_beta)           - Beta tiles (Wt tiles)
  c_8  (cb_eps)            - Epsilon tile (1 tile)
  c_16 (cb_normalized)     - Normalized output tiled (Wt tiles)
  c_17 (cb_out)            - RM output sticks (Wt tiles)
"""

import struct
from pathlib import Path
import ttnn


# Kernel files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


def _float_to_uint32(value: float) -> int:
    """Convert a float to its uint32 bit representation."""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for LayerNorm.

    Args:
        input_tensor: Input tensor 2D [H, W], ROW_MAJOR, bfloat16, on device
        output_tensor: Pre-allocated output tensor (on device)
        gamma: Optional scale parameter 1D [W]
        beta: Optional shift parameter 1D [W]
        eps: Variance stabilizer

    Returns:
        ProgramDescriptor ready for execution via ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    shape = input_tensor.shape
    H = shape[0]
    W = shape[1]
    TILE_WIDTH = 32
    TILE_HEIGHT = 32
    Wt = W // TILE_WIDTH  # Width in tiles

    has_gamma = gamma is not None
    has_beta = beta is not None

    # For ROW_MAJOR input: page = one stick = W * element_size bytes
    # bfloat16 = 2 bytes per element
    _dtype_to_bytes = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 1}
    element_size = _dtype_to_bytes.get(input_tensor.dtype, 2)
    stick_size = W * element_size

    # Tile size for bfloat16 (32x32 tile)
    tile_size = ttnn.tile_size(input_tensor.dtype)

    # Number of tile-rows (each tile-row = 32 sticks)
    num_tile_rows = H // TILE_HEIGHT

    device = input_tensor.device()

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    # Use single core for initial implementation (simplest correct path)
    # The design specifies split_blocks_for_tilize pattern but single core
    # is sufficient for stub validation.
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Single core gets all tile-rows
    nblocks_per_core = num_tile_rows
    num_sticks = H  # All sticks for this core
    start_stick_id = 0

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices as specified in design
    CB_IN = 0
    CB_TILIZED = 1
    CB_REDUCE_SCALER = 2
    CB_MEAN = 3
    CB_CENTERED = 4
    CB_VAR = 5
    CB_GAMMA = 6
    CB_BETA = 7
    CB_EPS = 8
    CB_NORMALIZED = 16
    CB_OUT = 17

    dtype = input_tensor.dtype

    # RM CBs use tile_size as page_size (tile-sized pages for RM data)
    # Tile CBs use tile_size as page_size

    cbs = []

    # c_0: cb_in - RM input sticks, Wt pages (tile-sized)
    cbs.append(
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

    # c_1: cb_tilized - Tilized input, Wt tiles
    cbs.append(
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

    # c_2: cb_reduce_scaler - 1/W reduce scaler, 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_REDUCE_SCALER,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # c_3: cb_mean - Row mean, 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
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

    # c_4: cb_centered - x - mean, Wt tiles
    cbs.append(
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

    # c_5: cb_var - Row variance, 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
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

    # c_6: cb_gamma - Gamma tiles, Wt tiles (only if gamma provided)
    gamma_pages = Wt if has_gamma else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=gamma_pages * tile_size,
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

    # c_7: cb_beta - Beta tiles, Wt tiles (only if beta provided)
    beta_pages = Wt if has_beta else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=beta_pages * tile_size,
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

    # c_8: cb_eps - Epsilon tile, 1 tile
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
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

    # c_16: cb_normalized - Normalized output (tiled), Wt tiles
    cbs.append(
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

    # c_17: cb_out - RM output sticks, Wt pages (tile-sized)
    cbs.append(
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
    # Compile-time args: stick_size, has_gamma, has_beta, TensorAccessorArgs(input),
    #                     [TensorAccessorArgs(gamma)], [TensorAccessorArgs(beta)]
    reader_ct_args = [
        stick_size,
        int(has_gamma),
        int(has_beta),
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # NOTE: gamma/beta use InterleavedAddrGen in the kernel (not TensorAccessor),
    # so no TensorAccessorArgs needed for them in compile-time args.

    # Runtime args: src_addr, num_sticks, Wt, start_stick_id, gamma_addr, beta_addr, eps_value
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0
    eps_uint32 = _float_to_uint32(eps)

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_sticks,
        Wt,
        start_stick_id,
        gamma_addr,
        beta_addr,
        eps_uint32,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layernorm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Compile-time args: Wt, nblocks_per_core, has_gamma, has_beta
    compute_ct_args = [
        Wt,
        nblocks_per_core,
        int(has_gamma),
        int(has_beta),
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layernorm_compute.cpp"),
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
    # Compile-time args: stick_size, TensorAccessorArgs(output)
    writer_ct_args = [stick_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Runtime args: dst_addr, num_sticks, Wt, start_stick_id
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_sticks,
        Wt,
        start_stick_id,
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layernorm_writer.cpp"),
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
