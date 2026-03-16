# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
matmul_sc - Program Descriptor

Defines CBs, kernels, and runtime args for single-core tiled matmul C = A x B.

CB layout:
  cb_in0 = 0   : A tiles (2 pages, double-buffer)
  cb_in1 = 1   : B tiles (2 pages, double-buffer)
  cb_out = 16  : C tiles (2 pages, double-buffer)

Single core: (0, 0)
"""

from pathlib import Path
import ttnn

# Kernel files live in the kernels/ subdirectory (paths relative to tt-metal base)
_OP_DIR = Path(__file__).parent
_KERNEL_DIR = _OP_DIR / "kernels"

# Relative path prefix from tt-metal root for kernel source paths
_TT_METAL_ROOT = Path(__file__).parent.parent.parent.parent.parent  # ttnn/ttnn/operations/matmul_sc -> tt-metal

# CB indices
CB_IN0 = 0
CB_IN1 = 1
CB_OUT = 16


def create_program_descriptor(
    input_a: ttnn.Tensor,
    input_b: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
) -> ttnn.ProgramDescriptor:
    """
    Create ProgramDescriptor for matmul_sc.

    Args:
        input_a: Matrix A [M, K] on device
        input_b: Matrix B [K, N] on device
        output_tensor: Pre-allocated output C [M, N] on device

    Returns:
        ProgramDescriptor for ttnn.generic_op
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    tile_size = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes for bf16 32x32 tile

    # Tile counts from shapes
    M = input_a.shape[0]
    K = input_a.shape[1]
    N = input_b.shape[1]

    Mt = M // 32  # tile rows of A and C
    Kt = K // 32  # inner dimension tiles
    Nt = N // 32  # tile columns of B and C
    batch = 1  # rank-2 tensors: single batch

    # ========== 2. CORE GRID ==========
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # cb_in0: A tiles (2 pages for double-buffering)
    cb_in0_descriptor = ttnn.CBDescriptor(
        total_size=2 * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_IN0,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # cb_in1: B tiles (2 pages for double-buffering)
    cb_in1_descriptor = ttnn.CBDescriptor(
        total_size=2 * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_IN1,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # cb_out: C tiles (2 pages for double-buffering)
    cb_out_descriptor = ttnn.CBDescriptor(
        total_size=2 * tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUT,
                data_format=ttnn.bfloat16,
                page_size=tile_size,
            )
        ],
    )

    # ========== 4. KERNEL DESCRIPTORS ==========
    kernel_source_reader = str(_KERNEL_DIR / "matmul_sc_reader.cpp")
    kernel_source_compute = str(_KERNEL_DIR / "matmul_sc_compute.cpp")
    kernel_source_writer = str(_KERNEL_DIR / "matmul_sc_writer.cpp")

    # --- Reader kernel ---
    # Named compile-time args: cb_in0=0, cb_in1=1
    # Positional compile-time args: TensorAccessorArgs(A) chained with TensorAccessorArgs(B)
    reader_ct_args = [CB_IN0, CB_IN1]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_b).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_a.buffer_address(),  # in0_addr
        input_b.buffer_address(),  # in1_addr
        Mt,  # tile rows of A/C
        Kt,  # inner dimension tiles
        Nt,  # tile columns of B/C
        batch,  # always 1 for rank-2
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=kernel_source_reader,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # Named compile-time args: cb_out=16
    # Positional compile-time args: TensorAccessorArgs(C)
    writer_ct_args = [CB_OUT]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),  # out_addr
        Mt,  # tile rows of C
        Nt,  # tile columns of C
        batch,  # always 1 for rank-2
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=kernel_source_writer,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    # Named compile-time args: cb_in0=0, cb_in1=1, cb_out=16
    # No positional compile-time args for compute
    compute_ct_args = [CB_IN0, CB_IN1, CB_OUT]

    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = [
        Mt,  # tile rows of C
        Kt,  # inner dimension tiles
        Nt,  # tile columns of C
        batch,  # always 1 for rank-2
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=kernel_source_compute,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            dst_full_sync_en=True,
        ),
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_in0_descriptor, cb_in1_descriptor, cb_out_descriptor],
    )
