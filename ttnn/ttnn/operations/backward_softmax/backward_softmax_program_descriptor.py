# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for backward_softmax.

Implements the two-pass streaming VJP described in op_design.md:

    Pass 1 (per block): mul(dy, y) -> cb_prod  → accumulate_reduce_block -> cb_sum
    Pass 2 (per block): sub<SCALAR>(dy, cb_sum) -> cb_centered → mul(y, cb_centered) -> cb_grad_input

Single-core for Phase 0. Each lane (a row for dim=-1, or a column for dim=-2)
is processed end-to-end on one core; the lane loop runs sequentially.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def _pick_block_size(reduce_tiles: int, requested: int | None = None) -> int:
    """
    Choose BLOCK_SIZE so that BLOCK_SIZE divides reduce_tiles, BLOCK_SIZE <= 8.
    Default selection: largest divisor of `reduce_tiles` that is <= 8.
    """
    if requested is not None:
        if reduce_tiles % requested != 0:
            raise ValueError(
                f"backward_softmax: BLOCK_SIZE={requested} does not divide " f"reduce-dim tiles={reduce_tiles}"
            )
        return requested
    upper = min(8, reduce_tiles)
    for candidate in range(upper, 0, -1):
        if reduce_tiles % candidate == 0:
            return candidate
    return 1


def create_program_descriptor(
    grad_output: ttnn.Tensor,
    output: ttnn.Tensor,
    grad_input: ttnn.Tensor,
    *,
    dim: int = -1,
    block_size: int | None = None,
) -> ttnn.ProgramDescriptor:
    # ---- Shape & tile geometry ----
    shape = list(grad_output.shape)
    N, C, H, W = shape
    Ht = H // TILE_DIM
    Wt = W // TILE_DIM
    NC = N * C

    dim_is_w = dim == -1
    if dim_is_w:
        reduce_dim_tiles = Wt
        total_lanes = NC * Ht
    else:  # dim == -2
        reduce_dim_tiles = Ht
        total_lanes = NC * Wt

    BLOCK_SIZE = _pick_block_size(reduce_dim_tiles, block_size)
    NUM_BLOCKS = reduce_dim_tiles // BLOCK_SIZE

    # ---- Page sizes ----
    grad_output_page_size = grad_output.buffer_page_size()
    output_page_size = output.buffer_page_size()
    grad_input_page_size = grad_input.buffer_page_size()

    # Scaler tile is bfloat16 (2048 B) regardless of data dtype — the reduce
    # LLK accepts bfloat16 scaler when the input tiles are float32 with
    # data-format reconfig handled by the helpers.
    scaler_page_size = ttnn.tile_size(ttnn.bfloat16)
    scaler_dtype = ttnn.bfloat16

    # ---- Single-core layout ----
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # ---- CB indices ----
    CB_GRAD_OUTPUT = 0
    CB_OUTPUT = 1
    CB_SCALER = 2
    CB_GRAD_INPUT = 16
    CB_PROD = 24
    CB_SUM = 25
    CB_CENTERED = 26

    # ---- CB sizes (in pages) ----
    # cb_grad_output: streamed lockstep with cb_output by the reader. In pass
    # 1 mul consumes dy and y per-tile in lockstep, so 2 pages is enough. In
    # pass 2 sub consumes dy per-tile, so 2 pages is also enough for dy.
    cb_grad_output_pages = 2
    # cb_output: in pass 2, sub consumes ONLY dy (BLOCK_SIZE tiles per block);
    # y is not consumed until the subsequent mul. Because the reader pushes dy
    # and y in lockstep, y must have room for the full block while sub drains
    # dy. Sizing to 2*BLOCK_SIZE both unblocks the lockstep and adds
    # double-buffer headroom across blocks.
    cb_output_pages = 2 * BLOCK_SIZE
    cb_scaler_pages = 1  # one persistent scaler tile
    cb_grad_input_pages = 2  # double buffer
    # Sequential helpers (mul → accumulate_reduce_block) cannot pipeline. Size
    # cb_prod / cb_centered to a full block + headroom.
    cb_prod_pages = 2 * BLOCK_SIZE
    cb_sum_pages = 2  # 1 logical tile + double-buffer headroom
    cb_centered_pages = 2 * BLOCK_SIZE

    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_grad_output_pages * grad_output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_GRAD_OUTPUT,
                    data_format=grad_output.dtype,
                    page_size=grad_output_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_output_pages * output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT,
                    data_format=output.dtype,
                    page_size=output_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_scaler_pages * scaler_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=scaler_dtype,
                    page_size=scaler_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_grad_input_pages * grad_input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_GRAD_INPUT,
                    data_format=grad_input.dtype,
                    page_size=grad_input_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_prod_pages * grad_input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_PROD,
                    data_format=grad_output.dtype,
                    page_size=grad_input_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_sum_pages * grad_input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SUM,
                    data_format=grad_output.dtype,
                    page_size=grad_input_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_centered_pages * grad_input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=grad_output.dtype,
                    page_size=grad_input_page_size,
                )
            ],
        ),
    ]

    # ---- Compile-time args (shared across kernels where applicable) ----
    DIM_IS_W = 1 if dim_is_w else 0

    # Reader CT args: BLOCK_SIZE, NUM_BLOCKS, DIM_IS_W, Ht, Wt, num_lanes, then accessors.
    reader_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        DIM_IS_W,
        Ht,
        Wt,
        total_lanes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(grad_output).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    # Writer CT args: BLOCK_SIZE, NUM_BLOCKS, DIM_IS_W, Ht, Wt, num_lanes, then accessors.
    writer_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        DIM_IS_W,
        Ht,
        Wt,
        total_lanes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(grad_input).get_compile_time_args())

    # Compute CT args: BLOCK_SIZE, NUM_BLOCKS, DIM_IS_W, num_lanes.
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        DIM_IS_W,
        total_lanes,
    ]

    # ---- Runtime args (single core) ----
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        grad_output.buffer_address(),
        output.buffer_address(),
        0,  # start_lane
    ]

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        grad_input.buffer_address(),
        0,  # start_lane
    ]

    # ---- Kernels ----
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "backward_softmax_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "backward_softmax_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "backward_softmax_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
