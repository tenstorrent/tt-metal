# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for backward_softmax.

Implements the two-pass streaming VJP described in op_design.md, with a
shape-aware **input-buffering strategy** chosen at descriptor time
(Refinement 2):

    Pass 1: mul(dy, y) -> cb_prod -> reduce(SUM) -> cb_sum
    Pass 2: sub<COL/ROW>(dy, cb_sum) -> cb_centered -> mul(y, cb_centered) -> cb_grad_input

Multi-core (Refinement 1): the embarrassingly-parallel lanes (one tile-row for
``dim=-1``, one tile-column for ``dim=-2``) are distributed across the full
``device.compute_with_storage_grid_size()`` grid via
``ttnn.split_work_to_cores``.

Input-buffering strategy (Refinement 2): Phase 0 streamed each input tile
twice from DRAM (once per pass). When the per-core L1 budget allows, the
reader instead reads each tile **once** per lane and the compute kernel reads
from L1 across both passes. Three strategies, picked deterministically at
descriptor time:

  1. ``WHOLE_ROW_DB`` — input CBs sized ``2 * reduce_dim_tiles`` pages.
     The reader prefetches lane N+1 into the second half of the CB while
     compute is mid-lane on lane N. DRAM reads halved AND read latency
     overlapped with compute.
  2. ``WHOLE_ROW_SB`` — input CBs sized ``reduce_dim_tiles`` pages. DRAM
     reads halved; reader and compute alternate per lane (no cross-lane
     overlap).
  3. ``PER_TILE_STREAM`` — Phase-0 behavior. Input CBs sized to a small
     constant (2 pages). Each tile read twice. Fallback for shapes whose
     reduce dimension is too large to cache in L1.

The picker prefers DB → SB → PER_TILE in that order. See ``_pick_strategy``
for the L1-budget arithmetic.

Whole-row strategies (1 and 2) use the same kernel logic — only CB sizes
differ. The kernel branches on a compile-time ``STRATEGY_IS_WHOLE_ROW`` flag
to choose between the whole-row path (one bulk mul + one bulk reduce + one
bulk sub + one bulk mul per lane) and the per-tile streaming path (the
Phase-0 block-loop).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


# -----------------------------------------------------------------------------
# Input-buffering strategy
# -----------------------------------------------------------------------------
#
# Conservative per-core L1 budget for ALL circular buffers, on Wormhole B0.
# Hardware L1 is ~1.5 MB per Tensix with ~1 MB available after firmware and
# dispatch overhead. We budget 700 KB for CBs to leave 300+ KB of headroom
# for kernel stacks, JIT scratch, and any helper-allocated scratch — the
# verifier note in op_requirements.md flags that an off-by-2× CB-budget
# overestimate hangs the device at allocation time.
L1_CB_BUDGET_BYTES = 700 * 1024


class _Strategy:
    """Buffering strategies, in order of preference (smallest = best)."""

    WHOLE_ROW_DB = 1
    WHOLE_ROW_SB = 2
    PER_TILE_STREAM = 3


def _strategy_name(s: int) -> str:
    return {
        _Strategy.WHOLE_ROW_DB: "WHOLE_ROW_DB",
        _Strategy.WHOLE_ROW_SB: "WHOLE_ROW_SB",
        _Strategy.PER_TILE_STREAM: "PER_TILE_STREAM",
    }[s]


def _is_whole_row(s: int) -> bool:
    return s in (_Strategy.WHOLE_ROW_DB, _Strategy.WHOLE_ROW_SB)


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


def _strategy_l1_usage(
    strategy: int,
    reduce_dim_tiles: int,
    block_size: int,
    grad_output_page_size: int,
    output_page_size: int,
    grad_input_page_size: int,
    scaler_page_size: int,
) -> int:
    """
    Bytes used by all CBs on one core for the chosen strategy. Used both by
    the picker and (indirectly via tests) for asserting which strategy a
    given shape lands on.
    """
    # Sizes that do NOT scale with strategy.
    cb_scaler_bytes = 1 * scaler_page_size
    cb_sum_bytes = 2 * grad_output_page_size  # 1-tile result + double-buffer headroom
    cb_grad_input_bytes = 2 * grad_input_page_size  # double-buffer streaming to writer
    fixed_bytes = cb_scaler_bytes + cb_sum_bytes + cb_grad_input_bytes

    if strategy == _Strategy.WHOLE_ROW_DB:
        # Inputs double-buffered (2N each), intermediates sized to full row (N each).
        cb_grad_output_bytes = 2 * reduce_dim_tiles * grad_output_page_size
        cb_output_bytes = 2 * reduce_dim_tiles * output_page_size
        cb_prod_bytes = reduce_dim_tiles * grad_output_page_size
        cb_centered_bytes = reduce_dim_tiles * grad_output_page_size
    elif strategy == _Strategy.WHOLE_ROW_SB:
        # Inputs single-buffered (N each), intermediates sized to full row.
        cb_grad_output_bytes = reduce_dim_tiles * grad_output_page_size
        cb_output_bytes = reduce_dim_tiles * output_page_size
        cb_prod_bytes = reduce_dim_tiles * grad_output_page_size
        cb_centered_bytes = reduce_dim_tiles * grad_output_page_size
    else:  # PER_TILE_STREAM (Phase-0 sizing)
        cb_grad_output_bytes = 2 * grad_output_page_size
        cb_output_bytes = 2 * block_size * output_page_size
        cb_prod_bytes = 2 * block_size * grad_output_page_size
        cb_centered_bytes = 2 * block_size * grad_output_page_size

    return cb_grad_output_bytes + cb_output_bytes + cb_prod_bytes + cb_centered_bytes + fixed_bytes


def _pick_strategy(
    reduce_dim_tiles: int,
    block_size: int,
    grad_output_page_size: int,
    output_page_size: int,
    grad_input_page_size: int,
    scaler_page_size: int,
) -> int:
    """
    Deterministic strategy selection: pick the deepest (i.e. most caching)
    strategy whose per-core CB working set fits within ``L1_CB_BUDGET_BYTES``.
    Preference order: WHOLE_ROW_DB → WHOLE_ROW_SB → PER_TILE_STREAM.
    """
    for candidate in (_Strategy.WHOLE_ROW_DB, _Strategy.WHOLE_ROW_SB, _Strategy.PER_TILE_STREAM):
        bytes_needed = _strategy_l1_usage(
            candidate,
            reduce_dim_tiles,
            block_size,
            grad_output_page_size,
            output_page_size,
            grad_input_page_size,
            scaler_page_size,
        )
        if bytes_needed <= L1_CB_BUDGET_BYTES:
            return candidate
    # PER_TILE_STREAM is the unconditional fallback — even gigantic reduces
    # only need ~200 KB of CBs in that strategy, so we should never get here.
    return _Strategy.PER_TILE_STREAM


# Public function used by tests to introspect which strategy a given shape
# triggers. Tests assert on the *name* so the test set is stable across
# budget tweaks.
def pick_strategy_name(
    grad_output: ttnn.Tensor,
    *,
    dim: int = -1,
    block_size: int | None = None,
) -> str:
    shape = list(grad_output.shape)
    _, _, H, W = shape
    Ht = H // TILE_DIM
    Wt = W // TILE_DIM
    reduce_dim_tiles = Wt if dim == -1 else Ht
    bs = _pick_block_size(reduce_dim_tiles, block_size)
    grad_output_page_size = grad_output.buffer_page_size()
    scaler_page_size = ttnn.tile_size(ttnn.bfloat16)
    return _strategy_name(
        _pick_strategy(
            reduce_dim_tiles,
            bs,
            grad_output_page_size,
            grad_output_page_size,  # output dtype matches
            grad_output_page_size,  # grad_input dtype matches
            scaler_page_size,
        )
    )


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

    # ---- Strategy selection (Refinement 2) ----
    strategy = _pick_strategy(
        reduce_dim_tiles,
        BLOCK_SIZE,
        grad_output_page_size,
        output_page_size,
        grad_input_page_size,
        scaler_page_size,
    )
    strategy_is_whole_row = _is_whole_row(strategy)

    # ---- Multi-core work distribution ----
    # Lanes are embarrassingly parallel. Spread them across the full
    # compute-with-storage grid; `ttnn.split_work_to_cores` guarantees the
    # per-core lane counts differ by at most one. `all_cores` is the union of
    # cores that actually receive work, which is what every kernel + CB binds
    # to (cores outside this union are not part of the program).
    device = grad_output.device()
    grid_size = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        lanes_per_core_g1,
        lanes_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_lanes)

    core_grid = all_cores

    # ---- CB indices ----
    CB_GRAD_OUTPUT = 0
    CB_OUTPUT = 1
    CB_SCALER = 2
    CB_GRAD_INPUT = 16
    CB_PROD = 24
    CB_SUM = 25
    CB_CENTERED = 26

    # ---- CB sizes (in pages), strategy-dependent ----
    # Strategy 1 (WHOLE_ROW_DB):
    #   - Inputs sized 2*N pages: reader prefetches lane N+1 into the second
    #     half while compute is on lane N. Each input tile read from DRAM
    #     exactly once per output tile.
    # Strategy 2 (WHOLE_ROW_SB):
    #   - Inputs sized N pages: reader/compute alternate per lane. Each
    #     input tile read from DRAM exactly once per output tile.
    # Strategy 3 (PER_TILE_STREAM): Phase-0 sizing.
    #   - Inputs sized to a small constant (2 pages, double-buffer for
    #     streaming). Each input tile read TWICE from DRAM (once per pass).
    if strategy == _Strategy.WHOLE_ROW_DB:
        cb_grad_output_pages = 2 * reduce_dim_tiles
        cb_output_pages = 2 * reduce_dim_tiles
        cb_prod_pages = reduce_dim_tiles
        cb_centered_pages = reduce_dim_tiles
    elif strategy == _Strategy.WHOLE_ROW_SB:
        cb_grad_output_pages = reduce_dim_tiles
        cb_output_pages = reduce_dim_tiles
        cb_prod_pages = reduce_dim_tiles
        cb_centered_pages = reduce_dim_tiles
    else:  # PER_TILE_STREAM — Phase-0 sizing
        # cb_grad_output: streamed lockstep with cb_output by the reader. In
        # pass 1 mul consumes dy and y per-tile in lockstep, so 2 pages is
        # enough. In pass 2 sub consumes dy per-tile, so 2 pages is also
        # enough for dy.
        cb_grad_output_pages = 2
        # cb_output: in pass 2, sub consumes ONLY dy (BLOCK_SIZE tiles per
        # block); y is not consumed until the subsequent mul. Because the
        # reader pushes dy and y in lockstep, y must have room for the full
        # block while sub drains dy. Sizing to 2*BLOCK_SIZE both unblocks
        # the lockstep and adds double-buffer headroom across blocks.
        cb_output_pages = 2 * BLOCK_SIZE
        # Sequential helpers (mul → accumulate_reduce_block) cannot pipeline.
        # Size cb_prod / cb_centered to a full block + headroom.
        cb_prod_pages = 2 * BLOCK_SIZE
        cb_centered_pages = 2 * BLOCK_SIZE

    # CBs that do not scale with strategy.
    cb_scaler_pages = 1  # one persistent scaler tile
    cb_grad_input_pages = 2  # double buffer to writer
    cb_sum_pages = 2  # 1 logical tile + double-buffer headroom

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
    # Per-core `num_lanes` and `start_lane` are RT args (they differ per core
    # under multi-core distribution). Everything else stays compile-time.
    DIM_IS_W = 1 if dim_is_w else 0
    STRATEGY_IS_WHOLE_ROW = 1 if strategy_is_whole_row else 0

    # Reader CT args: BLOCK_SIZE, NUM_BLOCKS, DIM_IS_W, Ht, Wt,
    # STRATEGY_IS_WHOLE_ROW, then accessors.
    reader_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        DIM_IS_W,
        Ht,
        Wt,
        STRATEGY_IS_WHOLE_ROW,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(grad_output).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    # Writer CT args: writer is strategy-agnostic — push order from compute is
    # row-major (dim=-1) or column-major (dim=-2) in both strategies.
    writer_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        DIM_IS_W,
        Ht,
        Wt,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(grad_input).get_compile_time_args())

    # Compute CT args: BLOCK_SIZE, NUM_BLOCKS, DIM_IS_W, STRATEGY_IS_WHOLE_ROW.
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        DIM_IS_W,
        STRATEGY_IS_WHOLE_ROW,
    ]

    # ---- Runtime args (one set per core in `all_cores`) ----
    # Walk groups in the same order `split_work_to_cores` walks them:
    # group 1 first (more lanes per core), then group 2. The cumulative
    # `current_lane` is the running `start_lane` for the next core, so the
    # union of [start_lane, start_lane + num_lanes) ranges covers
    # [0, total_lanes) exactly once with no overlap.
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    current_lane = 0
    for group, lanes_per_core in (
        (core_group_1, lanes_per_core_g1),
        (core_group_2, lanes_per_core_g2),
    ):
        if lanes_per_core == 0:
            continue
        for core_range in group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        grad_output.buffer_address(),
                        output.buffer_address(),
                        current_lane,  # start_lane
                        lanes_per_core,  # num_lanes
                    ]
                    writer_rt_args[x][y] = [
                        grad_input.buffer_address(),
                        current_lane,  # start_lane
                        lanes_per_core,  # num_lanes
                    ]
                    compute_rt_args[x][y] = [
                        lanes_per_core,  # num_lanes
                    ]
                    current_lane += lanes_per_core

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
        runtime_args=compute_rt_args,
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
