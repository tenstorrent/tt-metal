# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for layer_norm_rm.

L1-budget-bounded chunked design:
  Pass A (per strip): reader streams the input strip; compute tilizes per
                      chunk and runs accumulate_reduce_block<SUM, REDUCE_ROW>
                      with a 1/W scaler into cb_mean.
  Pass B (per strip): reader re-streams the input; compute tilizes per chunk,
                      sub<COL>(x, mean) → cb_centered, square_in_place,
                      accumulate_reduce_block into cb_inv_std (= variance).
  Eps + rsqrt:        transform_in_place on cb_inv_std (var → 1/sqrt(var+eps)).
  Pass C (per strip): reader re-streams the input and optionally gamma/beta;
                      compute tilizes per chunk, sub<COL>(x, mean) → centered,
                      mul_in_place<COL>(centered, inv_std), optional
                      mul_in_place<ROW>(centered, gamma_tilized), optional
                      add_in_place<ROW>(centered, beta_tilized), untilize.
                      Writer drains cb_output_rm.

Three reader passes per strip is the L1-budget trade-off: every CB is bounded
by BLOCK_SIZE (or by a constant), so per-core L1 stays constant regardless of W.
"""

from pathlib import Path
import struct

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# BLOCK_SIZE cap. With fp32_dest_acc_en=True, half-sync DEST = 4 tiles. The
# helpers internally batch via DEST_AUTO_LIMIT, but BLOCK_SIZE <= 8 keeps the
# batch loops well-behaved. See op_design.md, "Work Distribution".
BLOCK_SIZE_CAP = 8


# ---------------------------------------------------------------------------
# CB index assignments (per repo convention: 0-7 input, 8-15 special,
# 16-23 output, 24-31 intermediate).
# ---------------------------------------------------------------------------

CB_INPUT_RM = 0  # reader  -> compute (RM input sticks, TILE granularity)
CB_GAMMA_RM = 1  # reader  -> compute (RM gamma, ROW granularity)
CB_BETA_RM = 2  # reader  -> compute (RM beta,  ROW granularity)
CB_SCALER = 8  # reader  -> compute (1/W fp32 scaler, persistent)
CB_OUTPUT_RM = 16  # compute -> writer  (untilized RM output, TILE pages)
CB_TILIZED_X = 24  # compute -> compute (tilized input chunk)
CB_CENTERED = 25  # compute intermediate (centered / normalized)
CB_MEAN = 26  # compute intermediate (per-row mean, 1 tile, persistent)
CB_INV_STD = 27  # compute intermediate (variance then 1/sqrt(var+eps))
CB_GAMMA_TILIZED = 28  # compute intermediate (tilized gamma chunk)
CB_BETA_TILIZED = 29  # compute intermediate (tilized beta chunk)


def _pick_block_size(Wt: int, cap: int = BLOCK_SIZE_CAP) -> int:
    """Largest divisor of Wt that is <= cap (falls back to 1)."""
    for candidate in range(min(cap, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def _fp32_bits(value: float) -> int:
    """Pack an fp32 value into its 32-bit integer bit representation."""
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    # ------- Tensor metadata -------
    shape = list(input_tensor.shape)
    W = shape[-1]
    H = shape[-2]
    NC = 1
    for d in shape[:-2]:
        NC *= d

    total_rows = NC * H  # one RM stick per row
    num_strips = total_rows // TILE_DIM
    Wt = W // TILE_DIM

    BLOCK_SIZE = _pick_block_size(Wt)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    # Bytes for one strip-chunk: BLOCK_SIZE tiles wide * 32 elements/tile * 4 B
    chunk_bytes = BLOCK_SIZE * TILE_DIM * 4

    # Input / output stick sizes (one RM stick per row, W*4 bytes).
    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()
    assert input_page_size == W * 4, f"input page size mismatch: {input_page_size} != {W*4}"
    assert output_page_size == W * 4, f"output page size mismatch: {output_page_size} != {W*4}"

    # CB page sizes.
    tile_size_fp32 = ttnn.tile_size(ttnn.float32)  # 4096
    rm_chunk_page_size = chunk_bytes  # row-mode CB page for gamma/beta

    has_gamma = gamma is not None
    has_beta = beta is not None
    epsilon_bits = _fp32_bits(epsilon)

    # ------- Core grid + work distribution -------
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    (
        num_cores_total,
        all_cores,
        core_group_1,
        core_group_2,
        strips_per_core_group_1,
        strips_per_core_group_2,
    ) = ttnn.split_work_to_cores(grid, num_strips)

    # ------- Circular buffer descriptors -------
    # Sizing rules (see op_design.md):
    #   * Reader->compute and compute->writer CBs are double-buffered.
    #   * Intermediate sequential CBs hold the full block (BLOCK_SIZE pages).
    #   * Per-strip persistent CBs (mean, inv_std) hold 1 tile.
    #   * Scaler CB holds 1 tile, pushed once at boot and never popped until end.

    cbs = [
        # cb_input_rm: 2 * BLOCK_SIZE pages of tile_size each (TILE-granularity reader).
        ttnn.CBDescriptor(
            total_size=2 * BLOCK_SIZE * tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_RM,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
        # cb_output_rm: 2 * BLOCK_SIZE pages of tile_size each.
        ttnn.CBDescriptor(
            total_size=2 * BLOCK_SIZE * tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_RM,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
        # cb_scaler: 1 tile, fp32 (preserves precision for fp32 input + fp32 dst).
        ttnn.CBDescriptor(
            total_size=tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
        # cb_tilized_x: BLOCK_SIZE tiles (sequential intermediate, full block).
        ttnn.CBDescriptor(
            total_size=BLOCK_SIZE * tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_TILIZED_X,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
        # cb_centered: BLOCK_SIZE tiles (sequential intermediate, full block).
        ttnn.CBDescriptor(
            total_size=BLOCK_SIZE * tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
        # cb_mean: 1 tile, fp32 (per-row mean, persistent across passes).
        ttnn.CBDescriptor(
            total_size=tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MEAN,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
        # cb_inv_std: 1 tile, fp32 (variance then 1/sqrt(var+eps)).
        ttnn.CBDescriptor(
            total_size=tile_size_fp32,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INV_STD,
                    data_format=ttnn.float32,
                    page_size=tile_size_fp32,
                )
            ],
        ),
    ]

    if has_gamma:
        # cb_gamma_rm: 2 pages of chunk_bytes (ROW-granularity reader push).
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * rm_chunk_page_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_RM,
                        data_format=ttnn.float32,
                        page_size=rm_chunk_page_size,
                    )
                ],
            )
        )
        # cb_gamma_tilized: BLOCK_SIZE tiles (sequential intermediate).
        cbs.append(
            ttnn.CBDescriptor(
                total_size=BLOCK_SIZE * tile_size_fp32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_TILIZED,
                        data_format=ttnn.float32,
                        page_size=tile_size_fp32,
                    )
                ],
            )
        )

    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * rm_chunk_page_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_RM,
                        data_format=ttnn.float32,
                        page_size=rm_chunk_page_size,
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=BLOCK_SIZE * tile_size_fp32,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_TILIZED,
                        data_format=ttnn.float32,
                        page_size=tile_size_fp32,
                    )
                ],
            )
        )

    # ------- Compile-time args -------
    # Reader CT args:
    #   [0] BLOCK_SIZE
    #   [1] NUM_BLOCKS
    #   [2] chunk_bytes
    #   [3] scaler_bits  (1.0f / W, fp32 bit pattern)
    #   [4] HAS_GAMMA    (0 or 1)
    #   [5] HAS_BETA     (0 or 1)
    #   [6..] TensorAccessorArgs(input_tensor)
    #   [..]  TensorAccessorArgs(gamma) — placeholder when absent
    #   [..]  TensorAccessorArgs(beta)  — placeholder when absent
    scaler_bits = _fp32_bits(1.0 / float(W))
    reader_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        chunk_bytes,
        scaler_bits,
        1 if has_gamma else 0,
        1 if has_beta else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    if has_gamma:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())
    if has_beta:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    # Writer CT args:
    #   [0] BLOCK_SIZE
    #   [1] NUM_BLOCKS
    #   [2] chunk_bytes
    #   [3..] TensorAccessorArgs(output_tensor)
    writer_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        chunk_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute CT args:
    #   [0] BLOCK_SIZE
    #   [1] NUM_BLOCKS
    #   [2] HAS_GAMMA
    #   [3] HAS_BETA
    #   [4] epsilon_bits (fp32 bit pattern)
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        1 if has_gamma else 0,
        1 if has_beta else 0,
        epsilon_bits,
    ]

    # ------- Runtime args (per-core) -------
    # Reader:  [input_addr, gamma_addr, beta_addr, num_strips_for_core, start_strip_id]
    # Writer:  [output_addr, num_strips_for_core, start_strip_id]
    # Compute: [num_strips_for_core]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0

    strip_cursor = 0
    for group_cores, strips_per_core in (
        (core_group_1, strips_per_core_group_1),
        (core_group_2, strips_per_core_group_2),
    ):
        if strips_per_core == 0:
            continue
        for core_range in group_cores.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_addr,
                        gamma_addr,
                        beta_addr,
                        strips_per_core,
                        strip_cursor,
                    ]
                    writer_rt_args[x][y] = [output_addr, strips_per_core, strip_cursor]
                    compute_rt_args[x][y] = [strips_per_core]
                    strip_cursor += strips_per_core

    assert strip_cursor == num_strips, (
        f"layer_norm_rm: dispatched {strip_cursor} strips, expected {num_strips} " f"(split_work_to_cores mismatch)"
    )

    # ------- Kernels -------
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
