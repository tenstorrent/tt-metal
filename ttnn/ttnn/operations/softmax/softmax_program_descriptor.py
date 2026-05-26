# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for the softmax operation.

Refinement 1 — L1-budget-bounded chunked design.
Refinement 4 — non-tile-aligned shapes (W % 32 != 0 or H % 32 != 0).

The Phase-0 kernel sized `cb_input_tiles` to `2 × reduce_dim_tiles` and `cb_exps`
to `reduce_dim_tiles`, which OOMs at wide reduce dims (Wt = 128 / W = 4096
gave ~2.6 MB on a 1.5 MB L1 budget). The new design's per-core CB footprint is
bounded by a constant `BLOCK_SIZE` instead of scaling with `reduce_dim_tiles`.

Refinement 4: when the reduce dim's logical size is not a multiple of 32, the
last tile in the reduce strip is partially valid. The dataflow helper
`calculate_and_prepare_partial_reduce_scalers<>` emits a (full, partial)
scaler-tile pair into both `cb_max_scaler` and `cb_sum_scaler`; the compute side
forwards `ReducePartialScaler::last_tile_at(1)` to `reduce<MAX>` (Pass 1) and
`accumulate_reduce_block<SUM>` (Pass 2 / Pass 1-of-unstable). The partial
scaler zeros out padded positions so they contribute neutrally — 0 for SUM, and
the GMPOOL LLK's pre-multiply-then-max consumes the same mask. Pass 3 (the
output mul) produces garbage in padded positions; the host-side
`ttnn.to_layout(..., ROW_MAJOR_LAYOUT)` (or any read-back path that respects
the logical shape) discards them.

Chunked algorithm (numeric_stable=True, 3 reader passes over `x`):
    Pass 1: reduce<MAX, REDUCE_DIM, WaitAndPopPerTile>(cb_input_tiles, ...) → cb_max
            (full strip streams through; CB only needs 2 tiles double-buffered)
    Pass 2: for b in 0..NUM_BLOCKS-1:
                sub<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
                    cb_input_tiles, cb_max, cb_centered_exp, BLOCK_SIZE_shape,
                    post_op=exp)
                accumulate_reduce_block<SUM, REDUCE_DIM>(
                    cb_centered_exp, cb_sum_scaler, cb_inv_sum, BLOCK_SIZE_shape,
                    b, NUM_BLOCKS, post_op_final=recip)
    Pass 3: for b in 0..NUM_BLOCKS-1:
                sub<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
                    cb_input_tiles, cb_max, cb_centered_exp, BLOCK_SIZE_shape,
                    post_op=exp)
                mul<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
                    cb_centered_exp, cb_inv_sum, cb_output_tiles, BLOCK_SIZE_shape)

Numeric_stable=False (2 reader passes):
    Pass 1: per-block sfpu_exp + accumulate_reduce_block<SUM> + recip on last
    Pass 2: per-block sfpu_exp + mul by cb_inv_sum → cb_output_tiles

MAX + REDUCE_ROW does not support `Accumulate::at` (LLK pack-reduce edge mask
drops the running accumulator — see reduce_helpers_compute.inl:181). We sidestep
the limitation by streaming the full reduce dim through a single
`reduce<MAX, WaitAndPopPerTile>` call: the helper holds DST across all reduce-dim
tiles internally, packing once at the end, so the input CB never needs to hold
more than 1 tile (we double-buffer to 2 for reader-pipelining headroom).

Work distribution: one work-item = one reduce strip (1 × Wt for dim=-1, Ht × 1
for dim=-2). Strips are split across the full compute_with_storage grid via
`ttnn.split_work_to_cores`. Per-core counts are passed via runtime args.
"""

from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# Largest BLOCK_SIZE to consider. The cap matters because BLOCK_SIZE tiles must
# fit in `cb_centered_exp` (the per-block intermediate). At cap=16 and
# fp32 tiles (4 KB each), cb_centered_exp tops out at 64 KB — leaves ~1.4 MB
# of L1 headroom. Increasing the cap shrinks NUM_BLOCKS (fewer reduce-init
# round trips) but grows the intermediate CB.
BLOCK_SIZE_CAP = 16


# CB index ranges:
#   0–7   input
#   8–15  special (scalers, constants)
#   16–23 output
#   24–31 intermediate
#
# All names below are semantic, per repo policy.
CB_INPUT_TILES = 0
CB_MAX_SCALER = 8
CB_SUM_SCALER = 9
CB_OUTPUT_TILES = 16
CB_MAX = 24
CB_INV_SUM = 25
CB_CENTERED_EXP = 26  # per-block intermediate for exp(x - max) (or exp(x) on the unstable path)


def _pick_block_size(reduce_dim_tiles: int, cap: int = BLOCK_SIZE_CAP) -> int:
    """Pick the largest BLOCK_SIZE ≤ cap that divides reduce_dim_tiles.

    Falls back to 1 if no divisor in [1, cap] divides — but every reduce_dim_tiles
    is divisible by 1, so the function always returns ≥ 1.
    """
    for candidate in range(min(cap, reduce_dim_tiles), 0, -1):
        if reduce_dim_tiles % candidate == 0:
            return candidate
    return 1


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    dim: int,
    numeric_stable: bool,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    # ------- Tensor metadata -------
    shape = list(input_tensor.shape)
    n, c, h, w = shape

    nc = n * c
    # Refinement 4: ceil division so non-aligned H/W produce the correct
    # *storage* tile count (TILE_LAYOUT pads each H/W up to a tile-aligned
    # boundary). The reader/writer index into Ht*Wt tiles per NC, regardless
    # of the logical (non-aligned) size; the partial scaler masks the unused
    # positions in the last reduce-dim tile.
    Ht = (h + TILE_DIM - 1) // TILE_DIM
    Wt = (w + TILE_DIM - 1) // TILE_DIM

    # Reduce-strip definition (see op_design.md, "Reduce-strip definition").
    # dim=-1 → strip = 1×Wt tiles, num_strips = NC*Ht, reduce_dim_tiles = Wt.
    # dim=-2 → strip = Ht×1 tiles, num_strips = NC*Wt, reduce_dim_tiles = Ht.
    if dim == -1:
        num_strips = nc * Ht
        reduce_dim_tiles = Wt
        reduce_dim_logical = w
    elif dim == -2:
        num_strips = nc * Wt
        reduce_dim_tiles = Ht
        reduce_dim_logical = h
    else:
        raise ValueError(f"softmax program descriptor: unsupported dim={dim}")

    # Refinement 4 — partial-scaler handling.
    # `partial` is the number of *valid* reduce-axis elements in the last tile
    # along the reduce dim (0 when tile-aligned). The reader/compute both gate
    # on `has_partial`; when nonzero, the scaler CBs each carry a (full,
    # partial) tile pair instead of a single full tile.
    partial = reduce_dim_logical % TILE_DIM
    has_partial = 1 if partial > 0 else 0

    BLOCK_SIZE = _pick_block_size(reduce_dim_tiles)
    NUM_BLOCKS = reduce_dim_tiles // BLOCK_SIZE

    # numeric_stable=True → 3 reader passes (max, sum/exp, mul). False → 2 (sum/exp, mul).
    num_input_passes = 3 if numeric_stable else 2

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    # Advisory deviation from op_design.md: scaler CB is fp32, not bf16.
    # The design table says bf16 for the scaler, but with bf16 the reduce
    # LLK multiplies fp32 SrcA by bf16 SrcB which downcasts the product to
    # bf16 precision (~3e-3 relative). At fp32 input precision this
    # produces softmax sums that miss 1.0 by ~2e-3 (atol=1e-3 in the test).
    # The scaler dataflow helper (`prepare_reduce_scaler`) explicitly
    # supports both Float16_b and Float32 formats, so making the CB fp32
    # preserves the full SrcA precision through the multiply-accumulate.
    # Same applies for bf16 input + fp32 dest acc — the upcast-then-multiply
    # through Float32 scaler matches the dest accumulator width.
    scaler_dtype = ttnn.float32
    scaler_page_size = ttnn.tile_size(scaler_dtype)  # fp32 tile = 4096 B

    # Refinement 2 — intermediate-CB format selection (per /numeric-formats-metal §4).
    #
    # cb_max, cb_inv_sum, cb_centered_exp park running-accumulator state across
    # phase boundaries (max across passes 2&3, inv_sum across pass 3, the per-block
    # exp(x-max) between sub+exp and the reduce/mul). When fp32_dest_acc_en=True we
    # keep these at Float32 regardless of input dtype — otherwise the dest-accumulator
    # fp32 gain is erased at the pack-to-CB boundary. When fp32_dest_acc_en=False we
    # let intermediates match the input dtype (dest is bf16-wide anyway).
    if compute_kernel_config.fp32_dest_acc_en:
        intermediate_dtype = ttnn.float32
        intermediate_page_size = ttnn.tile_size(intermediate_dtype)  # 4096 B for fp32
    else:
        intermediate_dtype = input_tensor.dtype
        intermediate_page_size = input_page_size

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
    #
    # All sizes below are BOUNDED BY A CONSTANT — no `reduce_dim_tiles` term.
    # That is the deliverable of /memory-budget-metal.
    #
    #   cb_input_tiles : 2                  → reader/compute double-buffer (per-tile streaming).
    #   cb_max_scaler  : 1                  → persistent, NoPop.
    #   cb_sum_scaler  : 1                  → persistent, NoPop.
    #   cb_output_tiles: 2                  → compute/writer double-buffer (per-tile streaming).
    #   cb_max         : 1                  → one tile per strip, held across passes 2 & 3.
    #   cb_inv_sum     : 1                  → one tile per strip, held across pass 3.
    #   cb_centered_exp: BLOCK_SIZE         → per-block intermediate (sub+exp output then reduce/mul input).
    #
    # cb_centered_exp must hold BLOCK_SIZE tiles because the sub helper pushes all
    # BLOCK_SIZE tiles before the downstream reduce/mul starts consuming (sub and
    # the next helper are sequential within compute; no real pipelining).

    cb_input_tiles_pages = 2
    cb_output_tiles_pages = 2
    cb_max_pages = 1
    cb_inv_sum_pages = 1
    cb_centered_exp_pages = BLOCK_SIZE

    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_input_tiles_pages * input_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_TILES,
                    data_format=input_tensor.dtype,
                    page_size=input_page_size,
                )
            ],
        ),
        # Refinement 4: when partial > 0 the reader emits TWO scaler tiles per
        # scaler CB (the full + partial pair); the compute side waits on
        # both. Size each scaler CB accordingly. Aligned shapes keep the
        # 1-tile sizing.
        ttnn.CBDescriptor(
            total_size=(2 if has_partial else 1) * scaler_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MAX_SCALER,
                    data_format=scaler_dtype,
                    page_size=scaler_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=(2 if has_partial else 1) * scaler_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SUM_SCALER,
                    data_format=scaler_dtype,
                    page_size=scaler_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_output_tiles_pages * output_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=output_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_max_pages * intermediate_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MAX,
                    data_format=intermediate_dtype,
                    page_size=intermediate_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_inv_sum_pages * intermediate_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INV_SUM,
                    data_format=intermediate_dtype,
                    page_size=intermediate_page_size,
                )
            ],
        ),
        ttnn.CBDescriptor(
            total_size=cb_centered_exp_pages * intermediate_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED_EXP,
                    data_format=intermediate_dtype,
                    page_size=intermediate_page_size,
                )
            ],
        ),
    ]

    # ------- Compile-time args (shared across all cores) -------
    # dim_is_row: 1 for dim=-1 (REDUCE_ROW + BroadcastDim::COL), 0 for dim=-2 (REDUCE_COL + ROW).
    dim_is_row = 1 if dim == -1 else 0
    numeric_stable_flag = 1 if numeric_stable else 0

    # Reader CT args:
    #   [0] dim_is_row             — selects pool-type/reduce-dim-aware scaler overload
    #   [1] Ht
    #   [2] Wt
    #   [3] reduce_dim_tiles
    #   [4] num_input_passes       — 3 for numeric_stable, 2 otherwise
    #   [5] partial                — Refinement 4: # valid reduce-axis elements in the last
    #                                tile (0 ⇒ tile-aligned, use single-tile scaler API;
    #                                1..31 ⇒ non-aligned, use partial scaler API)
    #   [6..] TensorAccessorArgs(input_tensor)
    reader_ct_args = [
        dim_is_row,
        Ht,
        Wt,
        reduce_dim_tiles,
        num_input_passes,
        partial,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Writer CT args:
    #   [0] dim_is_row
    #   [1] Ht
    #   [2] Wt
    #   [3] reduce_dim_tiles
    #   [4..] TensorAccessorArgs(output_tensor)
    writer_ct_args = [
        dim_is_row,
        Ht,
        Wt,
        reduce_dim_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute CT args:
    #   [0] dim_is_row
    #   [1] numeric_stable
    #   [2] Ht
    #   [3] Wt
    #   [4] reduce_dim_tiles
    #   [5] BLOCK_SIZE
    #   [6] NUM_BLOCKS
    #   [7] has_partial      — Refinement 4: 1 ⇒ partial-scaler routing on the last reduce-dim
    #                          iteration of reduce<MAX> (Pass 1) and accumulate_reduce_block<SUM>
    #                          (Pass 2). 0 keeps the aligned single-tile path.
    compute_ct_args = [
        dim_is_row,
        numeric_stable_flag,
        Ht,
        Wt,
        reduce_dim_tiles,
        BLOCK_SIZE,
        NUM_BLOCKS,
        has_partial,
    ]

    # ------- Runtime args (per-core: strip count + starting strip index) -------
    # Reader / Writer:
    #   [0] base buffer address
    #   [1] num_strips_for_this_core
    #   [2] start_strip_id
    #
    # Compute:
    #   [0] num_strips_for_this_core

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()

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
                    reader_rt_args[x][y] = [input_addr, strips_per_core, strip_cursor]
                    writer_rt_args[x][y] = [output_addr, strips_per_core, strip_cursor]
                    compute_rt_args[x][y] = [strips_per_core]
                    strip_cursor += strips_per_core

    assert (
        strip_cursor == num_strips
    ), f"softmax: dispatched {strip_cursor} strips, expected {num_strips} (split_work_to_cores mismatch)"

    # ------- Kernels -------
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "softmax_compute.cpp"),
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
