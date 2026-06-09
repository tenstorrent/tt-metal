# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for toy_variance.

Computes per-row population variance Var(x) = E[(x - E[x])^2] using a
two-pass streaming algorithm:
    Pass 1: stream x → cb_mean = E[x]            via accumulate_reduce
    Pass 2: stream x → cb_variance = E[(x-mean)^2] via sub<COL> + square + accumulate_reduce_block
The reduction axis is chunked into NUM_BLOCKS blocks of BLOCK_SIZE tiles each
so W can be arbitrarily wide (e.g. 32 x 64000) without exceeding L1.

Restricted to NC=1 (no leading batch tile rows beyond the H direction) since
binary_op_helpers' BinaryInputBlockShape carries no NC dimension.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def _pick_block_size(Wt: int, requested: int | None) -> int:
    """Pick a BLOCK_SIZE that divides Wt. Default to the largest divisor of Wt that is <= 8."""
    if requested is not None:
        if Wt % requested != 0:
            raise ValueError(f"toy_variance: BLOCK_SIZE={requested} does not divide Wt={Wt}")
        return requested
    for candidate in range(min(8, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    block_size: int | None = None,
    std_dev: bool = False,
) -> ttnn.ProgramDescriptor:
    input_shape = list(input_tensor.shape)
    origin_W = input_shape[-1]
    origin_H = input_shape[-2]

    NC = 1
    for d in input_shape[:-2]:
        NC *= d
    if NC != 1:
        raise ValueError(
            f"toy_variance: only NC=1 is supported (got leading dims producing NC={NC}); "
            "binary_op_helpers' BinaryInputBlockShape has no batch dimension."
        )

    # Tile counts cover the padded shape; the reduce uses a partial scaler on
    # the last W-tile to suppress contributions from padded positions. The H
    # direction can also be padded — those padded rows produce garbage outputs
    # that the caller is responsible for slicing off.
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    Ht = (origin_H + TILE_DIM - 1) // TILE_DIM

    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    BLOCK_SIZE = _pick_block_size(Wt, block_size)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    # Variance reduces over the *real* W (= origin_W). With scaler = 1/N built
    # into the reduce, SUM produces means directly. The partial scaler tile
    # zeros out the contributions of the (TILE_DIM - partial_w) padded
    # positions in the last W-tile, so origin_W is the correct N (not Wt*32).
    inv_N = 1.0 / float(origin_W)
    inv_N_bits = struct.unpack("I", struct.pack("f", inv_N))[0]

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()
    output_num_pages = output_tensor.buffer_num_pages()

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Circular Buffers ---
    CB_IN = 0
    CB_CENTERED_SQ = 1
    CB_SCALER = 2
    CB_MEAN = 3
    CB_VARIANCE = 4
    CB_OUT = 16

    # cb_in: per-tile streaming for both passes. Double-buffer one block of
    # work for reader/compute pipelining.
    tiles_per_block = Ht * BLOCK_SIZE
    cb_in_pages = 2 * tiles_per_block

    # cb_centered_sq: holds (x - mean)^2 tiles for one block. The sub helper
    # (PerTile output) pushes them sequentially; the streaming reduce pops
    # them one at a time. Both are sequential within compute, so a single
    # block's worth of pages is sufficient; 2x for headroom.
    cb_centered_sq_pages = 2 * tiles_per_block

    # cb_mean: persistent across all of pass 2 (WaitUpfrontNoPop). After pass
    # 1 holds Ht tiles; capacity must be >= Ht.
    cb_mean_pages = max(2 * Ht, 2)

    # cb_variance: streaming reduce accumulator for pass 2. Pop-1/push-1 per
    # ht per block, so capacity >= Ht. 2x for safety.
    cb_variance_pages = max(2 * Ht, 2)

    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_in_pages * input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=input_tensor.dtype, page_size=input_page_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_centered_sq_pages * input_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED_SQ, data_format=input_tensor.dtype, page_size=input_page_size
                )
            ],
        ),
        ttnn.CBDescriptor(
            # 2 tiles when has_partial_w (full + partial scaler), else 1.
            total_size=(2 if has_partial_w else 1) * ttnn.tile_size(ttnn.bfloat16),
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER, data_format=ttnn.bfloat16, page_size=ttnn.tile_size(ttnn.bfloat16)
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_mean_pages * output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MEAN, data_format=output_tensor.dtype, page_size=output_page_size
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_variance_pages * output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_VARIANCE, data_format=output_tensor.dtype, page_size=output_page_size
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * output_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=output_page_size
                )
            ],
        ),
    ]

    # --- Reader ---
    reader_ct_args = [
        Ht,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [input_tensor.buffer_address()]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = [output_num_pages]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address()]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    compute_ct_args = [
        Ht,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        1 if std_dev else 0,
        1 if has_partial_w else 0,
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
