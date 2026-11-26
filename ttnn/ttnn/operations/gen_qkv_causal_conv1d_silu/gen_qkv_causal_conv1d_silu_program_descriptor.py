# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gen_qkv_causal_conv1d_silu — plan selection and ProgramDescriptor.

One program: per core, one channel group (``block_ch_tiles`` channel tiles) and one contiguous
token range (``core_seq_tiles`` tile-rows), processed in blocks of ``block_rows`` tile-rows.

    reader  (NCRISC/NoC0): taps -> cb_tap_tiles (packed, tap p in row 2p);
                           RM:   halo + body sticks -> cb_rows slot (body rows [0, 32 - split_rows))
                           TILE: input tiles -> cb_input_tiles; cb_untilized_rows -> cb_rows slot
    compute (TRISCs):      [TILE: untilize] -> windowed tilize (4 shifted windows / tile-row)
                           -> sum_p window_p * rowbcast(tap_p) in DEST -> SiLU in DEST -> pack
    writer  (BRISC/NoC1):  cb_output_tiles -> q / k / v (per-tile routing of the channel axis);
                           RM split reader: body rows [32 - split_rows, 32) -> the reader's cb_rows slot

Every block knob (block_ch_tiles, block_rows, cores_per_group) and every buffer depth is defined
exactly once in this module; CB capacities, loop trip counts and grid sizing derive from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------------------------
# Op-math constants
# ---------------------------------------------------------------------------------------------
TAP_COUNT = 4  # the convolution has 4 taps
HALO_ROWS = TAP_COUNT - 1  # rows before a block's first token that the block reads
TILE_DIM = 32

# ---------------------------------------------------------------------------------------------
# Buffer-depth knobs (single source of truth; every CB capacity derives from them)
# ---------------------------------------------------------------------------------------------
ROWS_DEPTH = 2  # cb_rows slots: reader fills block b+1 while compute windows block b
OUT_DEPTH = 2  # cb_output_tiles tile-rows: writer drains row m while compute packs m+1
IN_DEPTH = 2  # TILE regime: cb_input_tiles tile-rows (reader fetch j+1 during untilize j)
UNTILIZED_DEPTH = 2  # TILE regime: cb_untilized_rows tile-rows (untilize j+1 during copy j)
SPLIT_HANDSHAKE_DEPTH = 2  # split reader: cb_split_requests / cb_split_done tokens in flight
SPLIT_HANDSHAKE_PAGE_BYTES = 16  # one uint32 L1 address per request (16 B: the L1 alignment)

# ---------------------------------------------------------------------------------------------
# Mechanism caps
# ---------------------------------------------------------------------------------------------
MAX_BLOCK_CH_TILES = 255  # fast tilize requires block_width_tiles < 256 (tilize_helpers.inl)

# ---------------------------------------------------------------------------------------------
# Block-rows target (op_design.md Perf lamps -> Overlap). block_rows is the coarsest balanced block
# that fits L1 *up to this target*. Measured on conv1d-004 (BH 11x10, device kernel time): target 3
# (the L1 bound) 152.9 us, 2: 151.5 us, 1: 130.7 us — a whole-block windowed tilize before the conv
# pass leaves the writer idle and delays the first output; tile-row granularity overlaps reader,
# compute and writer. conv1d-020: 75.2 us (5) vs 73.7 us (1). plan_limits(max_block_rows=...)
# replaces the target (pinned plans).
# ---------------------------------------------------------------------------------------------
OVERLAP_BLOCK_ROWS = 1
L1_BOUND_BYTES = 1_048_576  # BB-05: L1 the model leaves to the op on every core

# ---------------------------------------------------------------------------------------------
# Split reader (ROW_MAJOR regime; Refinement 2). Decode-size calls are DRAM-request-rate-bound per
# DRAM endpoint (~12.5 ns per request per bank port, BH: removing half of every bank's stick reads
# cut conv1d-001 8.35 -> 6.35 us; removing them on half the banks changed nothing). NoC1 reaches
# each bank through a different endpoint than NoC0, so the writer RISC (NoC1) reads body rows
# [32 - SPLIT_READER_ROWS, 32) of every tile-row while the reader (NoC0) reads the rest plus taps,
# history and halos. A multiple of 8 keeps both shares spread evenly over BH's 8 DRAM banks.
# Measured (BH 11x10, device kernel us, split_rows 0 / 16 / 24 / 32):
#   001 8.37 / 6.45 / 5.96 / 6.79    002 9.90 / 7.74 / 7.21    003 13.06 / 11.39 / 11.00
#   004 127.5 / 129.0 / 124.3 (8: 127.6; compute-bound)       020 73.7 / 62.1 / 59.5
#   006 (L1 input) 6.27 / - / 5.84   010 (L1 output) 10.43 / - / 7.54   011 (fp32) 14.44 / - / 12.46
# 24 balances the two ports on decode: NoC0 = 8 rows + taps + history ~ NoC1 = 24 rows per bank.
# plan_limits(split_rows=...) pins it (0 = the reader reads every row).
# ---------------------------------------------------------------------------------------------
SPLIT_READER_ROWS = 24

# ---------------------------------------------------------------------------------------------
# CB indices (semantic names; the numbers are only slots)
# ---------------------------------------------------------------------------------------------
CB_ROWS = 0  # row-major sticks: halo(3) + body(32*rows) per slot
CB_TAP_TILES = 1  # packed tap tiles, resident
CB_WINDOW_TILES = 2  # 4 shifted windows per tile-row, one block
CB_OUTPUT_TILES = 3  # activated output tiles -> writer
CB_INPUT_TILES = 4  # TILE regime: input tiles -> untilize
CB_UNTILIZED_ROWS = 5  # TILE regime: untilized tile-rows -> reader -> cb_rows
CB_SPLIT_REQUESTS = 6  # split reader: reader -> writer, the L1 destination of each tile-row
CB_SPLIT_DONE = 7  # split reader: writer -> reader, "the writer's rows of this tile-row landed"

REGIME_RM = "rm_staged_window"
REGIME_TILE = "tile_untilize_staged_window"


@dataclass(frozen=True)
class Plan:
    """The plan a call runs (BB-08: reportable through get_last_plan())."""

    regime: str
    block_ch_tiles: int
    block_rows: int
    num_ch_groups: int
    cores_per_group: int
    num_cores: int
    core_seq_tiles_max: int
    split_rows: int  # RM: body rows per tile-row read by the writer RISC on NoC1 (0: reader reads all)


def select_regime(layout) -> str:
    return REGIME_RM if layout == ttnn.ROW_MAJOR_LAYOUT else REGIME_TILE


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def cb_layout(
    block_ch_tiles: int,
    block_rows: int,
    regime: str,
    *,
    split_rows: int = 0,
    elem_bytes: int = 2,
    tile_bytes: int = 2048,
):
    """The single definition of every CB capacity: [(cb index, page bytes, pages)].

    Both the L1 budget check (footprint) and the program descriptor read it, so a change to a
    knob or a depth lands in exactly one place (l1_ledger.md: Capacity column).
    """
    stick_bytes = block_ch_tiles * TILE_DIM * elem_bytes
    layout = [
        (CB_ROWS, stick_bytes, ROWS_DEPTH * (TILE_DIM * block_rows + HALO_ROWS)),
        (CB_TAP_TILES, tile_bytes, block_ch_tiles),
        (CB_WINDOW_TILES, tile_bytes, TAP_COUNT * block_ch_tiles * block_rows),
        (CB_OUTPUT_TILES, tile_bytes, OUT_DEPTH * block_ch_tiles),
    ]
    if regime == REGIME_TILE:
        layout.append((CB_INPUT_TILES, tile_bytes, IN_DEPTH * block_ch_tiles))
        layout.append((CB_UNTILIZED_ROWS, tile_bytes, UNTILIZED_DEPTH * block_ch_tiles))
    if split_rows:
        layout.append((CB_SPLIT_REQUESTS, SPLIT_HANDSHAKE_PAGE_BYTES, SPLIT_HANDSHAKE_DEPTH))
        layout.append((CB_SPLIT_DONE, SPLIT_HANDSHAKE_PAGE_BYTES, SPLIT_HANDSHAKE_DEPTH))
    return layout


def footprint(
    block_ch_tiles: int,
    block_rows: int,
    regime: str,
    *,
    split_rows: int = 0,
    elem_bytes: int = 2,
    tile_bytes: int = 2048,
) -> int:
    """Per-core CB bytes of a plan (l1_ledger.md: Total per-core footprint)."""
    return sum(
        page_bytes * pages
        for _, page_bytes, pages in cb_layout(
            block_ch_tiles, block_rows, regime, split_rows=split_rows, elem_bytes=elem_bytes, tile_bytes=tile_bytes
        )
    )


def select_plan(
    tensor_seq_tiles: int,
    tensor_ch_tiles: int,
    grid_x: int,
    grid_y: int,
    regime: str,
    l1_budget: int = L1_BOUND_BYTES,
    *,
    max_block_rows: Optional[int] = None,
    max_cores_per_group: Optional[int] = None,
    split_rows: Optional[int] = None,
    elem_bytes: int = 2,
    tile_bytes: int = 2048,
) -> Plan:
    """Pure plan selection (op_design.md: Work Distribution -> select_plan).

    Candidates: every divisor w of tensor_ch_tiles with w <= 255, G = C_t / w groups <= N cores,
    footprint(w, 1) <= budget. Per candidate U = min(N // G, T_t, max_cores_per_group) token
    ranges per group; makespan = ceil(T_t / U) * w. Choose min makespan, ties -> larger w, then
    fewer cores. block_rows = the coarsest balanced block that fits the budget, up to
    max_block_rows (default OVERLAP_BLOCK_ROWS, the measured overlap optimum). split_rows = the
    writer's share of each tile-row's body reads (ROW_MAJOR only; default_split_rows), or the pin.
    """
    num_grid_cores = grid_x * grid_y
    cpg_cap = tensor_seq_tiles if max_cores_per_group is None else max(1, min(max_cores_per_group, tensor_seq_tiles))
    best = None
    for w in range(1, min(MAX_BLOCK_CH_TILES, tensor_ch_tiles) + 1):
        if tensor_ch_tiles % w:
            continue
        groups = tensor_ch_tiles // w
        if groups > num_grid_cores:
            continue
        if footprint(w, 1, regime, elem_bytes=elem_bytes, tile_bytes=tile_bytes) > l1_budget:
            continue
        ranges = min(num_grid_cores // groups, tensor_seq_tiles, cpg_cap)
        core_seq_tiles_max = _div_up(tensor_seq_tiles, ranges)
        # Fewest token ranges that keep the same makespan ("then fewer cores").
        ranges = _div_up(tensor_seq_tiles, core_seq_tiles_max)
        makespan = core_seq_tiles_max * w
        rank = (makespan, -w, groups * ranges)
        if best is None or rank < best[0]:
            best = (rank, w, groups, ranges, core_seq_tiles_max)
    if best is None:
        raise ValueError(
            f"gen_qkv_causal_conv1d_silu: no channel split of {tensor_ch_tiles} channel tiles fits "
            f"{num_grid_cores} cores within {l1_budget} B of L1 per core"
        )
    _, w, groups, ranges, core_seq_tiles_max = best

    block_rows_target = OVERLAP_BLOCK_ROWS if max_block_rows is None else max_block_rows
    row_cap = max(1, min(block_rows_target, core_seq_tiles_max))
    block_rows_fit = row_cap
    while (
        block_rows_fit > 1
        and footprint(w, block_rows_fit, regime, elem_bytes=elem_bytes, tile_bytes=tile_bytes) > l1_budget
    ):
        block_rows_fit -= 1
    num_blocks = _div_up(core_seq_tiles_max, block_rows_fit)
    block_rows = _div_up(core_seq_tiles_max, num_blocks)  # balanced: same block count, smallest ragged tail

    if regime != REGIME_RM:
        split = 0  # TILE regime: w tile reads per tile-row, not 32 stick reads — nothing to split
    elif split_rows is not None:
        split = max(0, min(TILE_DIM, split_rows))
    else:
        split = default_split_rows()

    return Plan(
        regime=regime,
        block_ch_tiles=w,
        block_rows=block_rows,
        num_ch_groups=groups,
        cores_per_group=ranges,
        num_cores=groups * ranges,
        core_seq_tiles_max=core_seq_tiles_max,
        split_rows=split,
    )


def default_split_rows() -> int:
    """ROW_MAJOR split reader: body rows per tile-row the writer RISC reads on NoC1 (measured best on
    every profiled ROW_MAJOR cell; see SPLIT_READER_ROWS)."""
    return SPLIT_READER_ROWS


def output_l1_share(tensor_seq_tiles: int, width_tiles, num_l1_banks: int, tile_bytes: int = 2048) -> int:
    """Per-core L1 that interleaved L1 outputs q/k/v occupy (subtracted from the CB budget)."""
    return sum(_div_up(tensor_seq_tiles * x_tiles, num_l1_banks) * tile_bytes for x_tiles in width_tiles)


# ---------------------------------------------------------------------------------------------
# Work distribution
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreWork:
    core: "ttnn.CoreCoord"
    ch_group: int
    core_seq_start: int  # first tile-row of this core's token range
    core_seq_tiles: int  # tile-rows in this core's token range


def assign_work(plan: Plan, tensor_seq_tiles: int, grid_x: int):
    """Core i = g * U + s (row-wise over the grid) gets channel group g and token range s."""
    base, rem = divmod(tensor_seq_tiles, plan.cores_per_group)
    work = []
    for g in range(plan.num_ch_groups):
        for s in range(plan.cores_per_group):
            i = g * plan.cores_per_group + s
            work.append(
                CoreWork(
                    core=ttnn.CoreCoord(i % grid_x, i // grid_x),
                    ch_group=g,
                    core_seq_start=s * base + min(s, rem),
                    core_seq_tiles=base + (1 if s < rem else 0),
                )
            )
    return work


def _row_wise_core_range_set(num_cores: int, grid_x: int):
    ranges = []
    full_rows, tail = divmod(num_cores, grid_x)
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if tail:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(tail - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index: int, page_size: int, num_pages: int, core_ranges, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


# Positions of the buffer addresses inside each kernel's per-core runtime args (rewritten on a
# program-cache hit; everything else in the descriptor is a function of the cache key).
READER_ADDR_SLOTS = 6  # input, history, tap0..tap3
WRITER_ADDR_SLOTS = 4  # q, k, v, input (split reader)
READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1
READER_NOC = ttnn.NOC.NOC_0  # ReaderConfigDescriptor lowers to RISCV_1 on NOC_0
TAPS_MCAST_PREFIX = "taps_mcast"  # named CT args taps_mcast_ct_offset / taps_mcast_rt_offset


def reader_addresses(input_tensor, history, taps):
    return [input_tensor.buffer_address(), history.buffer_address()] + [t.buffer_address() for t in taps]


def writer_addresses(input_tensor, outputs):
    return [t.buffer_address() for t in outputs] + [input_tensor.buffer_address()]


def create_program_descriptor(
    input_tensor,
    history,
    taps,
    outputs,
    widths,
    plan: Plan,
    *,
    grid_x: int,
    math_fidelity,
    fp32_dest_acc_en: bool,
    dst_full_sync_en: bool,
    input_column_offset: int = 0,
    page_channels: Optional[int] = None,
):
    """Build the single program. Returns (descriptor, work) — `work` lists the cores in the
    order their runtime args were written (used to rewrite addresses on a cache hit)."""
    dtype = input_tensor.dtype
    elem_bytes = input_tensor.element_size()
    tile_bytes = ttnn.tile_size(dtype)
    tensor_seq_tiles = int(input_tensor.shape[1]) // TILE_DIM
    # TILE regime page stride of one tile-row: the (padded) width in tiles. Column window: the
    # convolution channels are tiles [offset/32, offset/32 + C/32) of each tile-row.
    input_width_tiles = _div_up(int(input_tensor.shape[2]), TILE_DIM)
    width_tiles = [x // TILE_DIM for x in widths]
    w = plan.block_ch_tiles
    block_rows = plan.block_rows
    tile_input = plan.regime == REGIME_TILE

    work = assign_work(plan, tensor_seq_tiles, grid_x)
    core_ranges = _row_wise_core_range_set(plan.num_cores, grid_x)

    # ---- circular buffers (capacities: cb_layout, the same source footprint() budgets) ----
    cbs = [
        _cb(index, page_bytes, pages, core_ranges, dtype)
        for index, page_bytes, pages in cb_layout(
            w, block_rows, plan.regime, split_rows=plan.split_rows, elem_bytes=elem_bytes, tile_bytes=tile_bytes
        )
    ]

    # ---- reader ----
    history_page_channels = 0 if page_channels is None else int(page_channels)  # 0: token-paged [1, 3, C]
    reader_ct = [
        w,
        block_rows,
        int(tile_input),
        input_width_tiles,
        elem_bytes,
        tile_bytes,
        history_page_channels,
        plan.split_rows,
    ]
    for t in [input_tensor, history] + list(taps):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    # ---- writer (+ the split reader's share of the input rows) ----
    stick_bytes = w * TILE_DIM * elem_bytes
    writer_ct = [
        w,
        width_tiles[0],
        width_tiles[1],
        width_tiles[2],
        tile_bytes,
        block_rows,
        plan.split_rows,
        stick_bytes,
    ]
    for t in list(outputs) + [input_tensor]:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    # ---- compute ----
    compute_ct = [w, block_rows, int(tile_input)]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    read_addrs = reader_addresses(input_tensor, history, taps)
    write_addrs = writer_addresses(input_tensor, outputs)
    input_col_offset_tiles = input_column_offset // TILE_DIM
    for cw in work:
        ch_group_tile_start = cw.ch_group * w
        reader_rt[cw.core.x][cw.core.y] = read_addrs + [
            ch_group_tile_start,
            input_col_offset_tiles,
            cw.core_seq_start,
            cw.core_seq_tiles,
        ]
        writer_rt[cw.core.x][cw.core.y] = write_addrs + [
            ch_group_tile_start,
            cw.core_seq_start,
            cw.core_seq_tiles,
            input_col_offset_tiles,
        ]
        compute_rt[cw.core.x][cw.core.y] = [cw.core_seq_start, cw.core_seq_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gen_qkv_causal_conv1d_silu_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gen_qkv_causal_conv1d_silu_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_fidelity = math_fidelity
    compute_config.fp32_dest_acc_en = bool(fp32_dest_acc_en)
    compute_config.dst_full_sync_en = bool(dst_full_sync_en)
    compute_config.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gen_qkv_causal_conv1d_silu_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    # Regime tap_multicast_per_group: the taps are reuse-shared across a group's token ranges, so the
    # group's slot-0 core reads them from DRAM once and multicasts the packed tap tiles to the other
    # cores_per_group - 1 cores (groups are U consecutive row-major cores: <= 3 rectangles each).
    taps_mcast = ttnn.Mcast(
        input_tensor.device(),
        ttnn.McastConfig(noc=READER_NOC),
        core_ranges,
        plan.cores_per_group,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.RowMajor,
    )
    descriptor = ttnn.ProgramDescriptor(cbs=cbs)
    taps_mcast.attach(descriptor, TAPS_MCAST_PREFIX, [reader], 0)
    kernels = [None, None, compute]
    kernels[READER_KERNEL_INDEX] = reader
    kernels[WRITER_KERNEL_INDEX] = writer
    descriptor.kernels = kernels
    return descriptor, work


def rewrite_addresses(descriptor, work, input_tensor, history, taps, outputs):
    """Program-cache hit: only the buffer addresses change (BB-07)."""
    read_addrs = reader_addresses(input_tensor, history, taps)
    write_addrs = writer_addresses(input_tensor, outputs)
    reader_rt = descriptor.kernels[READER_KERNEL_INDEX].runtime_args
    writer_rt = descriptor.kernels[WRITER_KERNEL_INDEX].runtime_args
    for cw in work:
        args = reader_rt[cw.core.x][cw.core.y]
        for i, a in enumerate(read_addrs):
            args[i] = a
        args = writer_rt[cw.core.x][cw.core.y]
        for i, a in enumerate(write_addrs):
            args[i] = a
