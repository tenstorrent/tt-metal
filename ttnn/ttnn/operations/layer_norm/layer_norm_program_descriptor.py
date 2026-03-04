# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LayerNorm - Program Descriptor

Defines the ProgramDescriptor for the layer_norm operation:
  - Circular buffer layout following op_design.md
  - Reader kernel: tilize RM input sticks + load weight/bias
  - Compute kernel: mean, sub, square, variance, rsqrt, normalize, affine
  - Writer kernel: untilize output sticks back to DRAM

CB layout (from op_design.md):
  c_0   cb_in           : Wt RM pages (one tile-row of input)
  c_1   cb_eps          : 1 tile (scalar epsilon, program lifetime)
  c_2   cb_scaler       : 1 tile (reduce scaler = 1/W, program lifetime)
  c_3   cb_weight       : Wt tiles (gamma, program lifetime, optional)
  c_4   cb_bias         : Wt tiles (beta,  program lifetime, optional)
  c_16  cb_tilize_out   : Wt tiles (tilize output / compute input)
  c_17  cb_out          : Wt tiles (normalized output, pre-untilize)
  c_18  cb_untilize_out : Wt tile-sized pages (RM sticks after untilize)
  c_24  cb_mean         : 1 tile (row mean)
  c_25  cb_x_minus_mean : Wt tiles (x - mean intermediates)
  c_26  cb_sq           : Wt tiles ((x-mean)^2 intermediates)
  c_27  cb_var          : 1 tile (variance)
  c_28  cb_inv_std      : 1 tile (1/sqrt(var+eps))
  c_29  cb_norm         : Wt tiles (normalized x, pre-affine)

Kernel paths (relative to tt-metal repo root):
  ttnn/ttnn/operations/layer_norm/kernels/reader.cpp
  ttnn/ttnn/operations/layer_norm/kernels/compute.cpp
  ttnn/ttnn/operations/layer_norm/kernels/writer.cpp
"""

import struct
from pathlib import Path
import ttnn


# Path to kernels directory — paths are RELATIVE to the tt-metal repo root
# (as required by KernelDescriptor.kernel_source)
_REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_DIR = Path("ttnn/ttnn/operations/layer_norm/kernels")


def _kernel_path(name: str) -> str:
    """Return repo-root-relative path string for a kernel file."""
    return str(KERNEL_DIR / name)


def _float_as_uint32(f: float) -> int:
    """Bit-cast a Python float to uint32 (for passing to kernels as bit pattern)."""
    return struct.unpack("I", struct.pack("f", f))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    weight: ttnn.Tensor = None,
    bias: ttnn.Tensor = None,
    eps: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for the layer_norm operation.

    Args:
        input_tensor:  2D RM input (N, W), on device.
        output_tensor: Pre-allocated 2D RM output (N, W), on device.
        weight:        Optional gamma (W,) tensor, on device.
        bias:          Optional beta  (W,) tensor, on device.
        eps:           Epsilon for numerical stability.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op.
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    N = input_tensor.shape[0]
    W = input_tensor.shape[1]

    Ht = N // 32  # tile-rows
    Wt = W // 32  # tiles per row

    dtype = input_tensor.dtype

    # Compute element size from dtype (bytes per element).
    # Using a lookup since tensor.element_size() is not available in this build.
    _ELEM_SIZE = {
        ttnn.bfloat16: 2,
        ttnn.float32: 4,
        ttnn.uint16: 2,
        ttnn.uint32: 4,
        ttnn.int32: 4,
    }
    elem_size = _ELEM_SIZE.get(dtype, 2)

    # Stick size (one RM row in bytes)
    stick_size = W * elem_size

    # Tile size (one 32x32 tile in bytes).
    # tile.get_tile_size(dtype) is available in all builds.
    tile_size = input_tensor.tile.get_tile_size(dtype)

    # Sizes for CB allocations
    block_width_size = Wt * 32 * elem_size  # bytes per tile-row block (used in writer compile-time)
    output_stick_size = W * elem_size  # same as stick_size for matching dtype

    has_weight = 1 if weight is not None else 0
    has_bias = 1 if bias is not None else 0

    # ========== 2. CORE GRID (single core 0,0) ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # c_0  cb_in: Wt RM pages for one tile-row of input
    #   The input CB holds RM data for the tilize helper.
    #   Page size = tile_size (the tilize helper reads tile-sized RM pages).
    cb_in_descriptor = ttnn.CBDescriptor(
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

    # c_1  cb_eps: 1 tile (scalar epsilon, program lifetime)
    cb_eps_descriptor = ttnn.CBDescriptor(
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

    # c_2  cb_scaler: 1 tile (reduce scaler = 1/W, program lifetime)
    cb_scaler_descriptor = ttnn.CBDescriptor(
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

    # c_3  cb_weight: Wt tiles (gamma, program lifetime)
    cb_weight_descriptor = ttnn.CBDescriptor(
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

    # c_4  cb_bias: Wt tiles (beta, program lifetime)
    cb_bias_descriptor = ttnn.CBDescriptor(
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

    # c_16 cb_tilize_out: Wt tiles (tilize output)
    cb_tilize_out_descriptor = ttnn.CBDescriptor(
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

    # c_17 cb_out: Wt tiles (normalized output before untilize)
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=17,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_18 cb_untilize_out: Wt tile-sized pages (RM data after untilize)
    cb_untilize_out_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=18,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_24 cb_mean: 1 tile (row mean)
    cb_mean_descriptor = ttnn.CBDescriptor(
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

    # c_25 cb_x_minus_mean: Wt tiles (x - mean intermediates)
    cb_x_minus_mean_descriptor = ttnn.CBDescriptor(
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

    # c_26 cb_sq: Wt tiles ((x-mean)^2 intermediates)
    cb_sq_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=26,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_27 cb_var: 1 tile (variance)
    cb_var_descriptor = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=27,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_28 cb_inv_std: 1 tile (1/sqrt(var+eps))
    cb_inv_std_descriptor = ttnn.CBDescriptor(
        total_size=tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=28,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    # c_29 cb_norm: Wt tiles (normalized output, pre-affine)
    cb_norm_descriptor = ttnn.CBDescriptor(
        total_size=Wt * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=29,
                data_format=dtype,
                page_size=tile_size,
            )
        ],
    )

    cbs = [
        cb_in_descriptor,
        cb_eps_descriptor,
        cb_scaler_descriptor,
        cb_weight_descriptor,
        cb_bias_descriptor,
        cb_tilize_out_descriptor,
        cb_out_descriptor,
        cb_untilize_out_descriptor,
        cb_mean_descriptor,
        cb_x_minus_mean_descriptor,
        cb_sq_descriptor,
        cb_var_descriptor,
        cb_inv_std_descriptor,
        cb_norm_descriptor,
    ]

    # ========== 4. READER KERNEL ==========
    # Compile-time args (reader):
    #   [0]  stick_size         (bytes per input row)
    #   [1]  tile_size          (bytes per tile, for reading weight/bias)
    #   [2]  TensorAccessorArgs (src - input)
    #   [3]  TensorAccessorArgs (weight - real or dummy)
    #   [4]  TensorAccessorArgs (bias - real or dummy)
    # Always pass all 3 TensorAccessorArgs to keep offsets fixed.
    # When weight/bias is None, use input tensor's accessor as dummy (never read).
    weight_for_accessor = weight if weight is not None else input_tensor
    bias_for_accessor = bias if bias is not None else input_tensor
    reader_ct_args = [stick_size, tile_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight_for_accessor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(bias_for_accessor).get_compile_time_args())

    # Runtime args (reader):
    #   [0]  src_addr          (input buffer DRAM address)
    #   [1]  num_rows          (N)
    #   [2]  Wt               (tiles per row = W/32)
    #   [3]  block_width_size  (W * elem_size per tile-row block)
    #   [4]  has_weight
    #   [5]  has_bias
    #   [6]  weight_addr       (0 if no weight)
    #   [7]  bias_addr         (0 if no bias)
    #   [8]  eps               (bit-cast float)
    weight_addr = weight.buffer_address() if weight is not None else 0
    bias_addr = bias.buffer_address() if bias is not None else 0

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        N,
        Wt,
        block_width_size,
        has_weight,
        has_bias,
        weight_addr,
        bias_addr,
        _float_as_uint32(eps),
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=_kernel_path("reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ========== 5. WRITER KERNEL ==========
    # Compile-time args (writer):
    #   [0]  cb_id_out              (c_18 = 18)
    #   [1]  output_stick_size      (W * elem_size)
    #   [2]  tile_height            (32)
    #   [3]  Ht                     (N / 32)
    #   [4]  num_tiles_per_block    (Wt)
    #   [5]  block_width_size       (Wt * 32 * elem_size)
    #   [6+] TensorAccessorArgs (dst)
    writer_ct_args = [
        18,  # cb_id_out = c_18
        output_stick_size,
        32,  # tile_height
        Ht,
        Wt,
        block_width_size,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Runtime args (writer):
    #   [0]  dst_addr  (output buffer DRAM address)
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=_kernel_path("writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 6. COMPUTE KERNEL ==========
    # Compile-time args (compute):
    #   [0]  Ht          (N / 32)
    #   [1]  Wt          (W / 32)
    #   [2]  has_weight
    #   [3]  has_bias
    compute_ct_args = [Ht, Wt, has_weight, has_bias]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=_kernel_path("compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # ========== 7. ASSEMBLE PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
