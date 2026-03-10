# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Builds the ProgramDescriptor (circular buffers, kernel descriptors, runtime args)
for the row-major layer normalization operation.

Work unit: 1 block = 1 tile-row = 32 RM sticks of width W (Wt tiles).
Work is distributed across cores via split_work_to_cores.
"""

import struct
from pathlib import Path

import ttnn

# Kernel source paths -- relative to tt-metal repo root (FILE_PATH source type)
_OP_DIR = "ttnn/ttnn/operations/layer_norm_rm/kernels"
_READER_KERNEL = f"{_OP_DIR}/layer_norm_rm_reader.cpp"
_COMPUTE_KERNEL = f"{_OP_DIR}/layer_norm_rm_compute.cpp"
_WRITER_KERNEL = f"{_OP_DIR}/layer_norm_rm_writer.cpp"


def _float_to_bfloat16_packed(val: float) -> int:
    """Pack a float as a uint32 with bfloat16 in both halves."""
    float_bytes = struct.pack(">f", val)
    bf16 = struct.unpack(">H", float_bytes[:2])[0]
    return (bf16 << 16) | bf16


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
        input_tensor: Input RM tensor on device.
        output_tensor: Pre-allocated RM output tensor on device.
        gamma: Optional gamma tensor on device.
        beta: Optional beta tensor on device.
        epsilon: Numerical stability constant.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op.
    """
    # ========== 1. TENSOR METADATA ==========
    shape = input_tensor.shape
    N, C, H, W = shape[0], shape[1], shape[2], shape[3]
    Wt = W // 32  # tiles per row
    num_rows = N * C * H  # total RM sticks
    num_blocks = num_rows // 32  # blocks = tile-rows (32 sticks each)

    # Page sizes
    # For RM tensors the page is one stick: W * element_size bytes
    stick_size = W * input_tensor.element_size()  # W * 2 for bfloat16

    # Tile page size for compute CBs (bfloat16 tile = 32*32*2 = 2048 bytes)
    tile_size = ttnn.tile_size(input_tensor.dtype)

    has_gamma = gamma is not None
    has_beta = beta is not None

    # ========== 2. CORE GRID & WORK DISTRIBUTION ==========
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(compute_grid, num_blocks)

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # All CBs use tile_size pages. RM CBs treat Wt tiles as 32 sticks.
    #
    # CB layout from design doc:
    #   c_0  (cb_rm_in)            : Wt pages - RM sticks for tilize input
    #   c_1  (cb_tilized)          : Wt pages - tilized tiles
    #   c_2  (cb_reduce_scaler)    : 1 page  - 1/W scaler
    #   c_3  (cb_eps)              : 1 page  - epsilon constant
    #   c_4  (cb_mean)             : 1 page  - row mean
    #   c_5  (cb_centered)         : Wt pages - x - mean (persistent)
    #   c_6  (cb_centered_sq)      : Wt pages - centered^2
    #   c_7  (cb_var)              : 1 page  - row variance
    #   c_16 (cb_out_pre_untilize) : Wt pages - normalized tiles
    #   c_17 (cb_rm_out)           : Wt pages - untilized RM sticks
    #   c_24 (cb_inv_std)          : 1 page  - rsqrt(var+eps)
    #   c_25 (cb_gamma)            : Wt pages - gamma tiles
    #   c_26 (cb_beta)             : Wt pages - beta tiles

    dtype = input_tensor.dtype

    def _make_cb(buffer_index: int, num_pages: int) -> ttnn.CBDescriptor:
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=buffer_index,
                    data_format=dtype,
                    page_size=tile_size,
                )
            ],
        )

    cb_rm_in = _make_cb(0, Wt)
    cb_tilized = _make_cb(1, Wt)
    cb_reduce_scaler = _make_cb(2, 1)
    cb_eps = _make_cb(3, 1)
    cb_mean = _make_cb(4, 1)
    cb_centered = _make_cb(5, Wt)
    cb_centered_sq = _make_cb(6, Wt)
    cb_var = _make_cb(7, 1)
    cb_out_pre_untilize = _make_cb(16, Wt)
    cb_rm_out = _make_cb(17, Wt)
    cb_inv_std = _make_cb(24, 1)
    cb_gamma = _make_cb(25, Wt)
    cb_beta = _make_cb(26, Wt)

    all_cbs = [
        cb_rm_in,
        cb_tilized,
        cb_reduce_scaler,
        cb_eps,
        cb_mean,
        cb_centered,
        cb_centered_sq,
        cb_var,
        cb_out_pre_untilize,
        cb_rm_out,
        cb_inv_std,
        cb_gamma,
        cb_beta,
    ]

    # ========== 4. COMPILE-TIME ARGS ==========
    # Reader compile-time args layout:
    #   [0]: stick_size
    #   [1]: gamma_accessor_ct_start (0 if no gamma)
    #   [2]: beta_accessor_ct_start (0 if no beta)
    #   [3+]: TensorAccessorArgs(input)
    #   [3+N]: TensorAccessorArgs(gamma) if has_gamma
    #   [3+N+M]: TensorAccessorArgs(beta) if has_beta
    reader_ct_args = [stick_size, 0, 0]  # slots 0, 1, 2
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    if has_gamma:
        reader_ct_args[1] = len(reader_ct_args)  # gamma accessor starts here
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    if has_beta:
        reader_ct_args[2] = len(reader_ct_args)  # beta accessor starts here
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    # Compute compile-time args: [Wt, has_gamma, has_beta]
    compute_ct_args = [Wt, int(has_gamma), int(has_beta)]

    # Writer compile-time args: [stick_size, Wt, TensorAccessorArgs(output)...]
    writer_ct_args = [stick_size, Wt]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ========== 5. RUNTIME ARGS (per-core) ==========
    eps_packed = _float_to_bfloat16_packed(epsilon)

    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0

    reader_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    # Track active cores so we can set empty args for idle cores afterwards.
    active_cores = set()
    current_block = 0

    # Helper to set args for a core group
    def _set_core_group_args(core_group, blocks_per_core):
        nonlocal current_block
        for core_range in core_group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    active_cores.add((x, y))
                    start_stick = current_block * 32
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),  # src_addr
                        blocks_per_core,  # num_blocks
                        start_stick,  # start_stick_id
                        Wt,  # Wt
                        int(has_gamma),  # has_gamma
                        int(has_beta),  # has_beta
                        gamma_addr,  # gamma_addr
                        beta_addr,  # beta_addr
                        eps_packed,  # eps_packed
                    ]
                    compute_rt_args[x][y] = [
                        blocks_per_core,  # num_blocks
                    ]
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),  # dst_addr
                        blocks_per_core,  # num_blocks
                        start_stick,  # start_stick_id
                    ]
                    current_block += blocks_per_core

    _set_core_group_args(core_group_1, blocks_per_core_g1)
    if blocks_per_core_g2 > 0:
        _set_core_group_args(core_group_2, blocks_per_core_g2)

    # ========== 6. KERNEL DESCRIPTORS ==========
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ========== 7. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=all_cbs,
    )
