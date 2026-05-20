# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for layer_norm_rm.

**Refinement 2** — wide-W streaming reduce + multi-core distribution.

Each Tensix core processes a disjoint slice of the flattened tile-row space
(`Ht_local` tile-rows out of `total_tile_rows`).  Per tile-row, three streaming
passes run with the reduction axis chunked into NUM_BLOCKS blocks of BLOCK_SIZE
tiles each, sized so `cb_input_tiles`/`cb_centered`/`cb_output` are constant
in `Wt`.

    Pass 1: stream input  → mean   (accumulate_reduce<SUM, REDUCE_ROW>, scaler 1/W)
    Pass 2: stream input  → variance via per-block sub<COL>+square_in_place+
                            accumulate_reduce_block, then transform_in_place
                            to rsqrt(var+eps)
    Pass 3: stream input  → centered = (x-mean) * inv_std (* gamma) (+ beta)
                            drain per block via copy_tiles (TILE out) or
                            untilize (RM out)

ROW_MAJOR input: per Pass-3 block, the reader streams 32 partial-row sticks of
`BLOCK_W * elem_size` bytes (one W-slice per row); compute calls
`tilize<BLOCK_SIZE, cb_input_sticks, cb_input_tiles>(1, 32)` to convert each
slice to tiles.  TILE input streams tiles directly into cb_input_tiles.

Gamma/beta (if present) are tilized once per Pass-3 block via per-block
partial-row reads: the reader emits 1 partial gamma stick of
`BLOCK_W * elem_size` per Pass-3 block per row; compute tilizes into
BLOCK_SIZE gamma tiles for that block.  cb_gamma_tiles is sized at
BLOCK_SIZE so the L1 footprint is constant in Wt.

BLOCK_SIZE selection: largest divisor of Wt that is ≤ 8 (matching toy_variance).
When W is non-tile-aligned (has_partial_w), BLOCK_SIZE = Wt and NUM_BLOCKS = 1
(streaming-with-partial-W is out of scope for Refinement 2 — partial-W shapes
are small and already fit in L1).

The scaler CB holds either a single full-tile scaler (W tile-aligned) or a
full+partial scaler pair (W non-tile-aligned). reduce<> waits but never pops;
the kernel issues a final cb_pop_front at exit.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32
MAX_BLOCK_SIZE = 8


def _padded_row_bytes(W: int, elem_size: int) -> int:
    tile_row_bytes = TILE_DIM * elem_size
    return math.ceil((W * elem_size) / tile_row_bytes) * tile_row_bytes


def _bits_of_float(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def _pick_block_size(Wt: int, has_partial_w: bool) -> int:
    """Largest divisor of Wt that is <= MAX_BLOCK_SIZE.

    For non-tile-aligned W, force BLOCK_SIZE = Wt (single block) — streaming
    + partial-W combination is out of scope for Refinement 2; the small-W
    cases that need partial-W support already fit in L1.
    """
    if has_partial_w:
        return Wt
    for candidate in range(min(MAX_BLOCK_SIZE, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def _iter_cores(core_range_set: ttnn.CoreRangeSet):
    """Yield (x, y) for every core in a CoreRangeSet, in row-major order."""
    for r in core_range_set.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                yield x, y


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor: ttnn.Tensor | None,
    beta_tensor: ttnn.Tensor | None,
    output_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
) -> ttnn.ProgramDescriptor:
    # ----- Tensor metadata -----
    input_shape = list(input_tensor.shape)
    origin_W = int(input_shape[-1])
    elem_size = input_tensor.element_size()
    tile_size = ttnn.tile_size(input_tensor.dtype)

    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    # total_tile_rows = ceil(prod(shape[:-1]) / 32). prod over all leading dims
    # (including the inner H), flattened, padded up to a multiple of TILE_DIM.
    total_rows = 1
    for d in input_shape[:-1]:
        total_rows *= int(d)
    total_tile_rows = (total_rows + TILE_DIM - 1) // TILE_DIM

    is_rm_input = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    is_rm_output = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma_tensor is not None
    has_beta = beta_tensor is not None

    padded_row_bytes = _padded_row_bytes(origin_W, elem_size)
    input_row_bytes = origin_W * elem_size  # actual valid bytes per row in DRAM

    # ----- Streaming block geometry -----
    BLOCK_SIZE = _pick_block_size(Wt, has_partial_w)
    NUM_BLOCKS = Wt // BLOCK_SIZE
    BLOCK_W = BLOCK_SIZE * TILE_DIM  # elements per block-wide stick (last block may have partial valid data)
    block_row_bytes = BLOCK_W * elem_size  # bytes per partial-row stick (allocation size)

    # Per-block input bytes to read from DRAM. For non-last blocks: full BLOCK_W
    # bytes. For the last block when has_partial_w: only the trailing valid
    # bytes (origin_W - (NUM_BLOCKS-1) * BLOCK_W). The reader uses this for
    # partial-row reads to avoid reading past the row's valid DRAM region.
    if has_partial_w:
        # has_partial_w => NUM_BLOCKS == 1 in this refinement; the single block
        # covers all valid columns of the row (input_row_bytes is the right
        # size).
        last_block_input_bytes = input_row_bytes
    else:
        # All blocks read BLOCK_W elements per row (tile-aligned, no partial).
        last_block_input_bytes = block_row_bytes

    inv_W = 1.0 / float(origin_W)
    inv_W_bits = _bits_of_float(inv_W)
    eps_bits = _bits_of_float(epsilon)

    # ----- Multi-core work distribution -----
    # Spread `total_tile_rows` tile-row groups across the compute grid.
    device = input_tensor.device()
    grid_size = device.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        Ht_per_core_g1,
        Ht_per_core_g2,
    ) = ttnn.split_work_to_cores(full_grid, total_tile_rows)

    # ----- Circular buffer indices -----
    CB_INPUT_STICKS = 0  # RM input only
    CB_INPUT_TILES = 1
    CB_GAMMA_STICKS = 2  # HAS_GAMMA only
    CB_BETA_STICKS = 3  # HAS_BETA only
    CB_SCALER = 4
    CB_OUTPUT = 16
    CB_GAMMA_TILES = 24  # HAS_GAMMA only
    CB_BETA_TILES = 25  # HAS_BETA only
    CB_MEAN = 26
    CB_INV_STD = 27
    CB_CENTERED = 28

    cbs: list[ttnn.CBDescriptor] = []

    # cb_input_tiles — BLOCK_SIZE tiles per block, double-buffered (constant in Wt).
    cb_input_tiles_pages = 2 * BLOCK_SIZE
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb_input_tiles_pages * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    if is_rm_input:
        # RM input: per-block partial-row sticks of BLOCK_W * elem_size each.
        # Size at 32 pages (one tile-row's worth) so the tilize LLK can span
        # 32 rows safely — required even though we push 32 sticks per block.
        # Double-buffer for pipelining.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * TILE_DIM * block_row_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_STICKS,
                        data_format=input_tensor.dtype,
                        page_size=block_row_bytes,
                    )
                ],
            )
        )

    # cb_scaler — bfloat16 ALWAYS (per reduce_helpers_dataflow.hpp).
    cb_scaler_pages = 2 if has_partial_w else 1
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb_scaler_pages * scaler_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=scaler_tile_size,
                )
            ],
        )
    )

    # cb_output — page_size is ALWAYS tile_size. The untilize helper outputs
    # tile-sized pages. For RM output the writer reads tile-sized L1 chunks
    # and writes partial-row sticks to DRAM. Sized at 2 * BLOCK_SIZE for
    # streaming drain.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * BLOCK_SIZE * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT,
                    data_format=output_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    # cb_mean, cb_inv_std — 1 tile each (per-tile-row scalar held across passes;
    # 2-page floor for double-buffer headroom).
    for cb_idx in (CB_MEAN, CB_INV_STD):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_idx,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # cb_centered — BLOCK_SIZE tiles, sized for one block's in-place chain.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * BLOCK_SIZE * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=input_tensor.dtype,
                    page_size=tile_size,
                )
            ],
        )
    )

    if has_gamma:
        # cb_gamma_sticks — per-block partial-row sticks. Size at 32 pages
        # (tile-row capacity) so the tilize LLK's 32-row read is safe even
        # though we only push 1 stick per block.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=TILE_DIM * block_row_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_STICKS,
                        data_format=input_tensor.dtype,
                        page_size=block_row_bytes,
                    )
                ],
            )
        )
        # cb_gamma_tiles — BLOCK_SIZE tiles per block (consumed and re-tilized
        # per Pass-3 block).  Double-buffer for pipelining.
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * BLOCK_SIZE * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA_TILES,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=TILE_DIM * block_row_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_STICKS,
                        data_format=input_tensor.dtype,
                        page_size=block_row_bytes,
                    )
                ],
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * BLOCK_SIZE * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA_TILES,
                        data_format=input_tensor.dtype,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # ----- Reader kernel -----
    # CT args: scalar args first, then TensorAccessorArgs at the end.
    reader_ct_args: list[int] = [
        Wt,  # 0
        BLOCK_SIZE,  # 1
        NUM_BLOCKS,  # 2
        block_row_bytes,  # 3 — per-partial-stick L1 size = BLOCK_W * elem_size
        last_block_input_bytes,  # 4 — DRAM read size for the last block
        padded_row_bytes,  # 5 — full row size (for gamma/beta DRAM addressing of the unpadded slice)
        input_row_bytes,  # 6
        elem_size,  # 7
        inv_W_bits,  # 8
        1 if has_partial_w else 0,  # 9
        partial_w if has_partial_w else 0,  # 10
        1 if is_rm_input else 0,  # 11
        1 if has_gamma else 0,  # 12
        1 if has_beta else 0,  # 13
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args()
        if gamma_tensor is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args()
        if beta_tensor is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # Per-core RT args: input_addr, gamma_addr, beta_addr, start_tile_row, Ht_local
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    work_groups = [
        (core_group_1, Ht_per_core_g1),
        (core_group_2, Ht_per_core_g2),
    ]

    next_start_tile_row = 0
    for core_group, work_per_core in work_groups:
        if work_per_core == 0:
            continue
        for x, y in _iter_cores(core_group):
            reader_rt_args[x][y] = [
                input_tensor.buffer_address(),
                gamma_tensor.buffer_address() if gamma_tensor is not None else 0,
                beta_tensor.buffer_address() if beta_tensor is not None else 0,
                next_start_tile_row,
                work_per_core,
            ]
            writer_rt_args[x][y] = [
                output_tensor.buffer_address(),
                next_start_tile_row,
                work_per_core,
            ]
            compute_rt_args[x][y] = [work_per_core]
            next_start_tile_row += work_per_core

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- Writer kernel -----
    writer_ct_args: list[int] = [
        Wt,  # 0
        BLOCK_SIZE,  # 1
        NUM_BLOCKS,  # 2
        block_row_bytes,  # 3
        padded_row_bytes,  # 4
        input_row_bytes,  # 5 — output_row_bytes == input_row_bytes
        last_block_input_bytes,  # 6 — last-block valid bytes (== output last-block valid bytes)
        elem_size,  # 7
        1 if has_partial_w else 0,  # 8
        1 if is_rm_output else 0,  # 9
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ----- Compute kernel -----
    compute_ct_args: list[int] = [
        Wt,  # 0
        BLOCK_SIZE,  # 1
        NUM_BLOCKS,  # 2
        1 if has_partial_w else 0,  # 3
        1 if is_rm_input else 0,  # 4
        1 if is_rm_output else 0,  # 5
        1 if has_gamma else 0,  # 6
        1 if has_beta else 0,  # 7
        eps_bits,  # 8
    ]

    compute_config = compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor()

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
