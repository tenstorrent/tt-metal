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
                           split writes: the reader writes WRITE_SHARE of every 8 output tiles on NoC0

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
# Split writes (perf tournament round 1, split_writes). The reader RISC (NoC0) writes output tile
# (r, cg) iff ((r + cg) * WRITE_SHARE) % 8 < WRITE_SHARE; the writer RISC (NoC1) writes the rest, so
# the DRAM writes use both NoCs' link sets. The NoC1-only write path is strongly position-unfair
# (writes-only ablation on conv1d-004: per-core ends 13 .. 74 us, bottom-right last).
# Measured (BH 11x10, device kernel us, WRITE_SHARE 0 -> 4):
#   004 (library body) 123.8 -> 123.6 (flat, compute-paced); 004 quartic body 149.5 -> 130.2 (median)
#   003 9.62 -> 9.12   006 5.61 -> 5.35   011 11.73 -> 11.14 (medians of 3); 002 005 019 020 010 flat;
#   001 6.04 -> 6.26 (+3.6%, w = 1: one tile per core, nothing to balance, the NoC0 write + token is the tail)
#   writes-only 004: 74.4 -> 68.2 (share 2: 61.2, 6: 96.9) — NoC0-heavy shares starve the top-left cores.
# 0 = the writer writes every tile (the previous sequence).
# ---------------------------------------------------------------------------------------------
WRITE_SHARE = 4
WRITE_DONE_PAGE_BYTES = 16  # one token per output group the reader wrote into
# Tokens in flight: bounds how many groups the reader may run ahead of the writer's pops. Any depth >= 1
# is deadlock-free (the reader polls for space and the writer consumes tokens in group order).
WRITE_DONE_DEPTH = 8

# bank_stagger (parked, measured null): rotate each core's body-row walk by its (channel group,
# token range) so the grid's first requests do not all land on one bank. Measured on 001 / 002 /
# 003: 5.87 / 6.81 / 10.02 -> 5.85 / 7.02 / 9.95 us (noise): every core queues its whole tile-row
# within ~1 us and each port is request-rate-bound, so issue order does not change the per-port
# totals. Kept as a live CT switch (reader + writer) for shapes where it may pay.
STAGGER_ROW_WALK = False

# ---------------------------------------------------------------------------------------------
# SiLU body (Refinement 3; Plan.silu_body -> compute CT arg). A 32-bit DEST always runs the library
# calculate_silu (fp32-accurate exp, two Newton steps); for a 16-bit DEST:
#   SILU_LIBRARY (0): the library body — exp_21f + 3-op float->int + reciprocal with a masked Newton
#     step, 26 SFPU instructions per row.
#   SILU_BF16_QUARTIC (1): silu_pack::silu_bf16_rows (compute kernel) — the same z = 2^floor(y)(1+f)
#     built by one shift, a quartic mantissa correction (relative error 1.1e-4 vs 1.7e-3), an unmasked
#     Newton step, two rows interleaved per step. Correctly rounded for ~99.4% of inputs vs 90-98%.
#     Measured SFPU cost of one pass over conv1d-004 (96 tiles / core, device-time difference of 2 and
#     3 passes, GQKV_SILU_REPEAT): 62.8 us (quartic, two rows) vs 79.2 us (library); a cubic one row
#     per step measured 66.4 (dependent-MAD stalls), a cubic two rows per step 58.8.
# default_silu_body picks the quartic body where it wins and the library body elsewhere (see there).
# ---------------------------------------------------------------------------------------------
SILU_LIBRARY = 0
SILU_BF16_QUARTIC = 1

# Per-tile STALLWAIT(STALL_SFPU, MATH) before each tile's SiLU on the packer (the library SFPU start
# issues it; redundant after the MATH_PACK semaphore wait — compute kernel, silu_from_pack). Measured
# null on conv1d-004 (library 129.9 -> 129.9 us, quartic family 149.8 -> 149.2): parked at True, the
# library start's exact sequence.
SILU_TILE_STALL = True


def dest_capacity_tiles(fp32_dest_acc_en: bool, dst_full_sync_en: bool) -> int:
    """Tiles one DEST acquire can hold (dest_helpers.hpp capacity table)."""
    return (16 if dst_full_sync_en else 8) // (2 if fp32_dest_acc_en else 1)


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
CB_WRITE_DONE = 8  # split writes: reader -> writer, "my writes of this output group are flushed"

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
    out_group_tiles: int  # output tiles per DEST acquire = per cb_output_tiles push = per writer wait
    silu_body: int = SILU_LIBRARY  # SFPU SiLU body for a 16-bit DEST (SILU_LIBRARY / SILU_BF16_QUARTIC)


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
    if WRITE_SHARE:
        layout.append((CB_WRITE_DONE, WRITE_DONE_PAGE_BYTES, WRITE_DONE_DEPTH))
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
    out_group_tiles: Optional[int] = None,
    silu_body: Optional[int] = None,
    dest_tiles: int = 8,
    fp32_dest_acc_en: bool = False,
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
    out_group_tiles = the output push quantum (default_out_group_tiles, or the pin), at most
    min(block_ch_tiles, dest_tiles). silu_body = the 16-bit-DEST SiLU body (default_silu_body, or the
    pin); a 32-bit DEST always runs the library body.
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
    group = default_out_group_tiles(w, dest_tiles, core_seq_tiles_max) if out_group_tiles is None else out_group_tiles
    group = max(1, min(group, w, dest_tiles))
    if fp32_dest_acc_en:
        body = SILU_LIBRARY
    else:
        body = default_silu_body(core_seq_tiles_max) if silu_body is None else silu_body

    return Plan(
        regime=regime,
        block_ch_tiles=w,
        block_rows=block_rows,
        num_ch_groups=groups,
        cores_per_group=ranges,
        num_cores=groups * ranges,
        core_seq_tiles_max=core_seq_tiles_max,
        split_rows=split,
        out_group_tiles=group,
        silu_body=body,
    )


def default_out_group_tiles(block_ch_tiles: int, dest_tiles: int, core_seq_tiles_max: int) -> int:
    """Output tiles per DEST acquire / cb_output_tiles push / writer wait (Refinement 2).

    One tile-row per core (decode-size calls): the whole output is a tail behind the last SiLU, so
    push every tile on its own — the writer stores tile c while the packer activates tile c+1, and
    the math thread's MACs for c+1 overlap SiLU(c) in the other DEST half. Several tile-rows per
    core: a full DEST per push. There the writer's flush per push is the throughput limit under
    DRAM write contention (one 2 KB write in flight per core at a 1-tile quantum).
    Measured (BH 11x10, device kernel us, quantum 1 vs full DEST / block):
      one row:   002 (w 2) 6.90 vs 7.32   003 (w 4) 9.89 vs 10.93   014 (w 3) 8.66 vs 9.82
                 016 (w 8) 15.64 vs 19.29 (2: 15.67, 4: 16.75)
      several:   004 (6 rows, w 16) 164.0 vs 123.4 (4: 149.4)     020 (5 rows, w 8) 69.1 vs 59.6
    """
    if core_seq_tiles_max == 1:
        return 1
    return min(block_ch_tiles, dest_tiles)


def default_silu_body(core_seq_tiles_max: int) -> int:
    """16-bit-DEST SiLU body (Refinement 3): the quartic body where a core owns one tile-row, the
    library body otherwise.

    One tile-row per core (decode-size calls): the activation of the core's only row is a tail on
    the critical path, so a cheaper SFPU pass is a direct win. Several tile-rows per core: the call is
    DRAM-bound once compute stops pacing it — with the quartic body the median core finishes ~9-13 us
    earlier on conv1d-004 (p50 115 -> 99-106 us), but the raised aggregate DRAM demand starves the
    cores with the longest NoC routes from the DRAM endpoints (a corner of the grid that moves with the
    split-reader ratio: NoC1-heavy split 24 -> bottom-right, NoC0-only split 0 -> top rows), whose tail
    sets the op time. The library body's pace keeps demand under DRAM capacity.
    Measured (BH 11x10, device kernel us, library -> quartic):
      one row:   001 5.89 -> 5.80   002 6.88 -> 6.56   003 9.84 -> 9.30   005 6.54 -> 6.41
                 006 5.39 -> 5.11   010 7.23 -> 6.85   012 15.48 -> 14.36   013 6.10 -> 5.80
                 014 8.60 -> 8.01   016 15.60 -> 14.33
      several:   004 129.6 (median of 3) -> ~150 (split 24; split 8: ~125-131 with a p100 tail of
                 136 us over a p50 of 103); 020 59.3 vs 60.9
    """
    return SILU_BF16_QUARTIC if core_seq_tiles_max == 1 else SILU_LIBRARY


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
READER_ADDR_SLOTS = 9  # input, history, tap0..tap3, q, k, v (split writes)
WRITER_ADDR_SLOTS = 4  # q, k, v, input (split reader)
READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1
# NoC of each data-movement kernel (the reader's also carries the tap multicast). The defaults are what
# ReaderConfigDescriptor / WriterConfigDescriptor lower to: reader RISCV_1 on NOC_0, writer RISCV_0 on NOC_1.
READER_NOC = ttnn.NOC.NOC_0
WRITER_NOC = ttnn.NOC.NOC_1


def _dm_config(processor, noc):
    config = ttnn.DataMovementConfigDescriptor()
    config.processor = processor
    config.noc = noc
    return config


TAPS_MCAST_PREFIX = "taps_mcast"  # named CT args taps_mcast_ct_offset / taps_mcast_rt_offset


def reader_addresses(input_tensor, history, taps, outputs):
    return (
        [input_tensor.buffer_address(), history.buffer_address()]
        + [t.buffer_address() for t in taps]
        + [t.buffer_address() for t in outputs]
    )


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
        int(STAGGER_ROW_WALK),
    ]
    for t in [input_tensor, history] + list(taps) + list(outputs):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_ct.extend([width_tiles[0], width_tiles[1], width_tiles[2], plan.out_group_tiles, WRITE_SHARE])
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
        plan.out_group_tiles,
        int(STAGGER_ROW_WALK),
    ]
    for t in list(outputs) + [input_tensor]:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_ct.append(WRITE_SHARE)
    # ---- compute ----
    compute_ct = [w, block_rows, int(tile_input), plan.out_group_tiles, plan.silu_body, int(SILU_TILE_STALL)]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    read_addrs = reader_addresses(input_tensor, history, taps, outputs)
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
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gen_qkv_causal_conv1d_silu_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_0, WRITER_NOC),
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
    read_addrs = reader_addresses(input_tensor, history, taps, outputs)
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
