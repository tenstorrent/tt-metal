# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm — realization of op_design.md's Blocking Model.

Every block factor, buffer depth and core-assignment named in the design is a
parameter here, defined exactly ONCE and derived from downstream:

    NUM_ROW_GROUPS        (g)  — independent row-group rectangles on the grid
    NUM_HIDDEN_SLICES     (s)  — cores splitting the reduced (hidden) axis inside a rect
    SLICE_HIDDEN_TILES    (S)  — the hidden block extent = the whole slice
    BLOCK_ROWS            (B)  — the row block extent (coarsest that fits L1)
    IN_CB_DEPTH / OUT_CB_DEPTH / RM_IN_DEPTH / RM_OUT_DEPTH — buffer-depth knobs
                               (IN_CB_DEPTH co-solves with BLOCK_ROWS in one
                                L1 ladder — see the knob's comment)
    HIDDEN_TILES_PER_CORE_FLOOR — hidden-granularity floor (bounds S from below)
    FANIN_BALANCE_K       — combine fan-in balance (bounds s from above)
    DM_CHUNK_TILES        — NoC tiles per barrier (reader AND writer)

CB page counts, loop trip counts and grid sizing are computed FROM those knobs;
no whole-op dimension (Wt, Rt) appears in any CB capacity.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# ---------------------------------------------------------------------------
# Knobs (single source of truth; see op_design.md §Parameters)
# ---------------------------------------------------------------------------
HIDDEN_TILES_PER_CORE_FLOOR = 4  # measured-fastest width-shard geometries land at 4-8
# Combine fan-in balance — the OTHER side of the hidden-split tradeoff.
#
# Splitting the hidden axis `s` ways cuts the per-core DRAM transfer (S = Wt/s
# tiles, so ~c2/s) but grows the cross-core combine, which costs ~c1*s: s partial
# tiles incast into ONE root, s gather atomics, a root reduce over s tiles, and a
# barrier across s cores.  `HIDDEN_TILES_PER_CORE_FLOOR` bounds only S; nothing
# bounded `s`, so the search maximized occupancy and ran the combine off the end.
#
# Minimizing c1*s + c2*Wt/s puts the optimum at s* = sqrt(c2*Wt/c1), i.e.
# proportional to sqrt(Wt) — NOT at a constant.  A constant cap was measured
# first and is wrong in exactly the way the model predicts: tuned to the wide
# single-row decode shapes it costs the tall-and-wide ones occupancy they need
# (a constant 32 regressed (1,1,64,12288) by 5.5%).
#
# Measured (Blackhole p150b, one fresh run per point, perf-target config), the
# wall is U-shaped in `s` and the minimum tracks sqrt(Wt) with k ~= 2.13:
#     Wt=224 (W=7168):  s=56 -> 11049 ns | s=32 ->  9732 | s=28 -> 9881 | s=14 -> 10901 | s=7 -> 14910
#     Wt=160 (W=5120):  s=40 ->  9316    | s=32 ->  8445 | s=27 -> 8377 | s=20 ->  8549 | s=10 -> 9846
#     Wt=72  (W=2304):  s=18 ->  6547    | s=9  ->  6707
#     Wt=32  (W=1024):  s=8  ->  5525    | s=4  ->  6173   (the floor already caps s at 8 here)
#   predicted s* = 2.13*sqrt(Wt):  Wt=224 -> 32, 160 -> 27, 72 -> 18, 32 -> 12 (censored to 8).
FANIN_BALANCE_K = 2.13


def _fanin_slice_cap(hidden_tiles: int) -> int:
    """The `s` above which the combine costs more than the transfer it saves."""
    return max(1, round(FANIN_BALANCE_K * (hidden_tiles**0.5)))


# Fuse `scale_block` + `apply_gamma_block` into one DEST window (compute kernel's
# GAMMA_FUSED).  The fused pass costs the same two FPU ops but halves the L1
# crossings — x*rsqrt is never packed out and read back — at the price of ONE
# up-front pass that expands gamma from its row-0 vector form to full tiles
# (DEST-reuse has no intra-tile broadcast).
#
# MEASURED and PARKED OFF (0 = never fuse), on the pinned BLOCK-shard prefill
# geometry (8192x1024, [1024,128] on 8x8 = 32 tile-rows per core, 4 blocks), which
# is the geometry with the most to gain:
#     unfused  74417 ns   |   fused (min_row_tiles=2)  78635 ns   +5.7%
# The saving is real (half the L1 crossings) but it is bought with a slower math
# datapath: `mul_reuse_dest_tiles` must copy DEST back into srcA before the second
# multiply, and that copy sits on the MATH thread between two dependent ops, while
# the unfused pair runs as two independent DEST windows that pipeline against the
# packer.  On this op the math thread binds and the L1 crossings do not, so the
# trade goes the wrong way.  The lever is CORRECT (the whole sharded suite passes
# with it on) and stays as a live knob: set it to 2 to fuse wherever a core owns
# more than one tile-row.  At 0 the program is byte-identical to Refinement 4.
GAMMA_FUSE_MIN_ROW_TILES = 0

# Tiles per DEST window on the streaming eltwise passes (compute's DEST_BLOCK_TILES).
# At 1 -- the Phase 0 value, implicit in a bare IterationShape -- every tile pays a
# whole tile_regs acquire/commit/wait/release round trip, which is a hard MATH<->PACK
# barrier per tile.  Batching lets the packer drain tile i while math runs tile i+1.
# The chain clamps the value to DEST_AUTO_LIMIT (4 with fp32 DEST accumulation, 8
# without) at runtime, and the kernel further clamps it to the largest DIVISOR of
# S (the grid walk groups row-wise, so a group wider than the row leaves a short
# tail whose per-tile output lifecycle under-pushes -- measured as a writer hang).
#
# MEASURED (sharded perf geometries, device kernel ns):
#     geometry                 block_size 1   batched
#     (1,1,8192,1024) BLOCK          74417     72063   -3.2%
#     (1,1,32,1024)   WIDTH           3819      3870   +1.3%
#     (1,1,32,2304)   WIDTH           4513      4644   +2.9%
#     (1,1,32,5120)   WIDTH           5690      5760   +1.2%
#     (1,1,32,7168)   WIDTH           5716      5823   +1.9%
# It wins only where a core owns many tile-rows: a batched window costs one extra
# fill/drain of the DEST pipeline per group, which a single-tile-row block (the
# whole decode regime -- one group in total) pays without ever amortizing.  Hence
# the row-tile gate below, which keeps that regime byte-identical to Refinement 4.
DEST_BLOCK_TILES = 8
DEST_BLOCK_MIN_ROW_TILES = 2

OUT_CB_DEPTH = 2  # cb_output_tiles double buffer (2.78x overlap, double_buffer/report.md)
RM_IN_DEPTH = 2  # cb_rm_stage_in  tile-row window
RM_OUT_DEPTH = 2  # cb_rm_stage_out tile-row window

# cb_input_tiles depth — the design's OVERLAP perf lamp, now a live knob.
#
# Phase 0 asserted this at 1 because the two in-place rewrites of x
# (mask_tail_block, scale_block) pack at `get_write_ptr(cb) + i*page`, and a
# compute thread that never pushes cb_input_tiles keeps its write pointer at the
# CB BASE for the kernel's life — so a deeper CB would have written x*r into the
# wrong half once the read window moved off base.
#
# Refinement 2 removed that coupling for the sharded path (`pack_base`, a runtime
# offset measured from the CB base) and Refinement 4 generalizes it: the pack
# index is `(block * BLOCK_TILES) % IN_CAPACITY_TILES`, which tracks the read
# window through a CB of ANY whole-block-multiple capacity.  So depth > 1 is now
# expressible, and the lamp is measurable in its correct form — "smaller
# block_rows + a SECOND input buffer", never "same block, deeper CB" (a deeper CB
# at the same block gives the reader nothing to prefetch INTO: with
# num_blocks == 1 there is no block b+1).  When IN_CB_DEPTH > 1 the row block is
# therefore capped so a core owns at least IN_CB_DEPTH blocks, and the ladder in
# `_plan` skips the rung entirely when it does not.
#
# MEASURED on the interleaved prefill profile (median of 3 fresh runs each,
# Blackhole p150b, target config, at DM_CHUNK_TILES = 32):
#     shape             depth 1   depth 2
#     (1,1,8192,1024)     91004     87836   -3.5%
#     (1,1,8192,2304)    198987    190696   -4.2%
#     (1,1,8192,5120)    411580    398782   -3.1%
#     (1,1,8192,7168)    561649    562318   +0.1%  (L1 cannot afford the second
#                                                   buffer -> ladder degrades to 1,
#                                                   so this row is the SAME program)
# The whole decode regime is one tile-row per core (max_rows_full == 1), so the
# guard above keeps it byte-identical to Refinement 3.
#
# A note on why the ladder degrades instead of forcing the rung: taking the
# second buffer by *coarsening past the budget* is a measured LOSS.  Widening
# l1_working_budget to the part's real 1.46 MB (which coarsens block_rows) cost
# +5.7% on (1,1,8192,5120) and +3.7% on (1,1,8192,7168) at depth 1 — a coarser
# block means a LONGER fully-serial read before compute starts, which is the
# opposite of what this profile wants.
IN_CB_DEPTH = 2

# Tiles per NoC barrier, reader AND writer alike (the ROW_MAJOR kernels convert
# this byte budget into sticks — see RM_CHUNK_STICKS there).  This bounds the
# bytes a core keeps IN FLIGHT: the reader issues this many page reads before it
# blocks on `noc_async_read_barrier`, so a small value drains the NoC to empty
# once per chunk and pays the DRAM round-trip latency serially.
#
# MEASURED on the interleaved prefill profile (Blackhole p150b, target config,
# device kernel ns, one fresh run per point):
#     DM_CHUNK_TILES     8      16      32      64     128
#     (1,1,8192,1024)  91787   92650   90320   90321   88619
#     (1,1,8192,2304) 206450  204793  198736  197887  194713
#     (1,1,8192,5120) 432974  418844  403463  412640  412017
#     (1,1,8192,7168) 580019  557670  559316  551158  561972
# 32 is the first value at the plateau and the only one that wins on every row;
# beyond it the rows disagree inside noise.  The decode regime is untouched: a
# decode block is 4-7 tiles, so it never reaches even the old chunk.
DM_CHUNK_TILES = 32

# L1 budget. `l1_size_per_core()` is not bound to Python; the fallback is a
# single named constant (conservative for WH/BH, both >= 1.4 MB usable).
L1_SIZE_PER_CORE_FALLBACK = 1024 * 1024
L1_RESERVE = 96 * 1024  # firmware / kernel text / semaphores headroom

# DRAM read-start granule (mirrors DRAM_ALIGN_BYTES in the reader kernel).
DRAM_ALIGN_BYTES = 64

# ---------------------------------------------------------------------------
# CB indices (semantic names; the numeric slot is just the buffer index)
# ---------------------------------------------------------------------------
CB_INPUT_TILES = 0
CB_GAMMA_TILES = 1
CB_SQ_PARTIALS = 2
# (index 3 intentionally unused: cb_slice_stat is elided — the collapse is fused
#  into the root's combine, so contributors ship raw cb_sq_partials tiles.)
CB_GATHERED_PARTIALS = 4
CB_RMS_BCAST = 5
CB_RMS_RECIP = 6
CB_SCALER = 7
CB_W_MASK = 8
CB_OUTPUT_TILES = 9
CB_RM_STAGE_IN = 10
CB_RM_STAGE_OUT = 11
CB_THREAD_SYNC = 12  # 1 page; carries no data, only the PACK->UNPACK edge for in-place handoffs
# ROW_MAJOR + sharded only: the caller's resident shards, bound zero-copy so the
# stick staging reads/writes are core-LOCAL L1 traffic instead of DRAM.  On the
# TILE + sharded path the shards ARE cb_input_tiles / cb_output_tiles (no extra
# index, no copy at all).
CB_SHARD_IN = 13
CB_SHARD_OUT = 14

SHARDED_MEMORY_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def is_sharded(tensor) -> bool:
    return tensor.memory_config().memory_layout in SHARDED_MEMORY_LAYOUTS


# ---------------------------------------------------------------------------
# Semaphore ids
# ---------------------------------------------------------------------------
SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER_PROGRESS = 2


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


# Block-float formats have no per-element byte size: 16 data elements share one
# exponent, so `Tensor.element_size()` raises for them ("datum for bfp2, bfp4,
# bfp8 is invalid", tt_backend_api_types.hpp).  The *_ELEM_BYTES compile-time
# args are consumed ONLY by the ROW_MAJOR stick paths (stick pitch, per-element
# byte offsets), and a block-float tensor is necessarily TILE-layout — there is
# no row-major encoding of a shared-exponent block.  So the value below is a
# never-dereferenced placeholder for that case, chosen as 1 so the reader's and
# writer's `RM_STICK_PITCH % 16 == 0` static_assert (evaluated on every program,
# not just ROW_MAJOR ones) still holds.
_BLOCK_FLOAT_ELEM_BYTES = 1


def _elem_bytes(tensor) -> int:
    """`tensor.element_size()`, defined for block-float formats too."""
    try:
        return tensor.element_size()
    except (ValueError, RuntimeError):
        return _BLOCK_FLOAT_ELEM_BYTES


def _l1_working_budget(device) -> int:
    query = getattr(device, "l1_size_per_core", None)
    total = query() if callable(query) else L1_SIZE_PER_CORE_FALLBACK
    return total - L1_RESERVE


# ---------------------------------------------------------------------------
# Tile geometry (alignment-aware, per-image)
# ---------------------------------------------------------------------------


def _tile_geometry(input_tensor):
    """(row_tiles, hidden_tiles, W, total_sticks) for either layout."""
    shape = list(input_tensor.shape)
    padded = list(input_tensor.padded_shape)
    w = shape[-1]
    hidden_tiles = _div_up(w, TILE_DIM)

    if input_tensor.layout == ttnn.TILE_LAYOUT:
        num_images = 1
        for d in padded[:-2]:
            num_images *= d
        row_tiles = num_images * _div_up(padded[-2], TILE_DIM)
        total_sticks = 0
        assert row_tiles * _div_up(padded[-1], TILE_DIM) == input_tensor.buffer_num_pages(), (
            f"rms_norm: tile-count cross-check failed: {row_tiles} x "
            f"{_div_up(padded[-1], TILE_DIM)} != {input_tensor.buffer_num_pages()}"
        )
    else:
        # ROW_MAJOR: rows are contiguous and may fold across image boundaries,
        # because the reduce runs along W only (rows never interact).
        #
        # Counted from the shape, NOT from buffer_num_pages(): a ROW_MAJOR page is
        # one *buffer* stick, and a width- or block-sharded buffer's stick is the
        # SHARD's width, so buffer_num_pages() there is (sticks x width-shards),
        # not the tensor's stick count.  RM carries no tile padding, so the padded
        # shape is the logical one.
        total_sticks = 1
        for d in padded[:-1]:
            total_sticks *= d
        row_tiles = _div_up(total_sticks, TILE_DIM)

    return row_tiles, hidden_tiles, w, total_sticks


# ---------------------------------------------------------------------------
# L1 fit predicate — mirrors l1_ledger.md's footprint_tiles() exactly
# ---------------------------------------------------------------------------


def _footprint_bytes(block_rows, slice_tiles, num_slices, *, is_row_major, has_gamma, bytes_, in_depth=1):
    b, s_, s = block_rows, slice_tiles, num_slices
    total = b * s_ * bytes_["in_tile"] * in_depth  # cb_input_tiles
    if has_gamma:
        total += s_ * bytes_["gamma_tile"]  # cb_gamma_tiles
    total += b * bytes_["stat_tile"]  # cb_sq_partials
    if s > 1:
        total += s * b * bytes_["stat_tile"]  # cb_gathered_partials
        total += b * bytes_["stat_tile"]  # cb_rms_bcast
    total += b * bytes_["stat_tile"]  # cb_rms_recip
    total += 3 * bytes_["bf16_tile"]  # cb_scaler + cb_w_mask + cb_thread_sync
    if is_row_major:
        total += RM_IN_DEPTH * s_ * bytes_["in_tile"]  # cb_rm_stage_in
        total += RM_OUT_DEPTH * s_ * bytes_["out_tile"]  # cb_rm_stage_out
    else:
        total += OUT_CB_DEPTH * s_ * bytes_["out_tile"]  # cb_output_tiles
    return total


def _footprint_bytes_sharded(
    block_rows,
    slice_tiles,
    num_slices,
    *,
    is_row_major,
    has_gamma,
    bytes_,
    rm_in_depth=RM_IN_DEPTH,
    rm_out_depth=RM_OUT_DEPTH,
):
    """Per-core CB bytes on the SHARDED path.

    Same ledger as `_footprint_bytes`, minus the two buffers that are no longer
    CB-heap allocations: on the TILE path `cb_input_tiles` / `cb_output_tiles`
    ARE the caller's resident shards (zero-copy, charged to the budget by the
    caller subtracting the shard bytes), so only the stat CBs and — on the
    ROW_MAJOR path — the tilize staging buffers are sized here.
    """
    b, s_, s = block_rows, slice_tiles, num_slices
    total = 0
    if is_row_major:
        # cb_input_tiles here is tilize's COMPUTE-side target, fed from the resident
        # shard through cb_rm_stage_in — no reader fills it, so the input-depth knob
        # does not apply (the staging CB carries its own rm_in_depth).  `_plan_sharded`
        # returns in_depth = 1 to match.
        total += b * s_ * bytes_["in_tile"]  # cb_input_tiles (tilize target)
        total += rm_in_depth * s_ * bytes_["in_tile"]  # cb_rm_stage_in
        total += rm_out_depth * s_ * bytes_["out_tile"]  # cb_rm_stage_out
    if has_gamma:
        total += s_ * bytes_["gamma_tile"]  # cb_gamma_tiles
    total += b * bytes_["stat_tile"]  # cb_sq_partials
    if s > 1:
        total += s * b * bytes_["stat_tile"]  # cb_gathered_partials
        total += b * bytes_["stat_tile"]  # cb_rms_bcast
    total += b * bytes_["stat_tile"]  # cb_rms_recip
    total += 3 * bytes_["bf16_tile"]  # cb_scaler + cb_w_mask + cb_thread_sync
    return total


# ---------------------------------------------------------------------------
# Work distribution / regime selection (op_design.md §Work Distribution)
# ---------------------------------------------------------------------------


def _rect_candidates(gx, gy):
    """Every (num_hidden_slices, rect_w, rect_h) a row-group rectangle can take.

    A row-group must be an exact rectangle because `Mcast2D` takes the bounding
    box of the core set as THE multicast rect: a non-rectangular group would
    broadcast a stat tile into cores that do not own those rows.  Any
    rect_w <= grid_x, rect_h <= grid_y tiles the grid by floor division; the
    leftover columns/rows simply hold no row-group.

    Pinning rect_w to grid_x (or to a divisor of it) is what starved the decode
    regime on an 11-wide grid: `s` was forced to a multiple of 11, so a shape
    wanting 56 slices could only take 11.  Searching rect_w x rect_h instead
    lets 56 land as 8 x 7.
    """
    return [(rect_w * rect_h, rect_w, rect_h) for rect_h in range(1, gy + 1) for rect_w in range(1, gx + 1)]


def _shard_core_list(spec):
    """The shard grid's cores in SHARD ORDER (shard i lives on cores[i])."""
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    return [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(spec.grid, None, row_wise)]


def _plan_sharded(device, input_tensor, *, has_gamma, bytes_):
    """Read the SAME partition the interleaved path searches for OFF the shard spec.

    All three flavours are the Phase 0 logical scheme with the geometry pinned by
    the caller (op_design.md §Lamps, "Physical shard placement"):

      HEIGHT -> one row-group per core, `num_hidden_slices == 1` (RowParallel:
                the reduce stays core-local, no combine at all).
      WIDTH  -> one row-group spanning every core, `num_hidden_slices == ncores`
                (exactly the Phase 0 gather-to-root + broadcast combine).
      BLOCK  -> the Phase 0 2D partition: one row-group per grid row.

    So `HIDDEN_TILES_PER_CORE_FLOOR` and the rect search do not apply here — the
    caller's shard spec IS the rect search's answer.
    """
    spec = input_tensor.memory_config().shard_spec
    if spec is None:
        raise RuntimeError("rms_norm: sharded input tensor without a shard_spec")
    cores = _shard_core_list(spec)
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])

    row_tiles, hidden_tiles, w, total_sticks = _tile_geometry(input_tensor)
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    slice_tiles = _div_up(shard_w, TILE_DIM)  # S — hidden tiles per core
    if is_row_major:
        # A ROW_MAJOR shard is `shard_h` sticks x `shard_w` elements; the core
        # tilizes its own sticks (rows never interact, so a shard boundary that
        # is not a multiple of 32 just means a zero-padded last tile-row).
        shard_rows = _div_up(shard_h, TILE_DIM)
        num_row_groups = _div_up(total_sticks, shard_h)
        num_hidden_slices = _div_up(w, shard_w)
    else:
        if shard_h % TILE_DIM or shard_w % TILE_DIM:
            raise RuntimeError(f"rms_norm: TILE shard {shard_h}x{shard_w} must be a multiple of 32x32")
        shard_rows = shard_h // TILE_DIM
        num_row_groups = _div_up(row_tiles, shard_rows)
        num_hidden_slices = _div_up(hidden_tiles, slice_tiles)

    if num_row_groups * num_hidden_slices != len(cores):
        raise RuntimeError(
            f"rms_norm: shard grid holds {len(cores)} cores but the shard shape "
            f"[{shard_h}, {shard_w}] implies {num_row_groups} x {num_hidden_slices} shards"
        )

    # The resident shards are charged against L1 up front (they are tensors, not
    # CB-heap allocations), then the coarsest row block that fits the remainder.
    shard_tiles = shard_rows * slice_tiles
    budget = _l1_working_budget(device) - shard_tiles * (bytes_["in_tile"] + bytes_["out_tile"])

    def _fits(b, in_depth, out_depth):
        return (
            _footprint_bytes_sharded(
                b,
                slice_tiles,
                num_hidden_slices,
                is_row_major=is_row_major,
                has_gamma=has_gamma,
                bytes_=bytes_,
                rm_in_depth=in_depth,
                rm_out_depth=out_depth,
            )
            <= budget
        )

    # Turn the coarse knobs first, the overlap knobs only if that is not enough.
    # A shard pins slice_hidden_tiles, so on the ROW_MAJOR path (three S-sized
    # staging buffers) the buffer-DEPTH knobs are the only remaining slack.
    #
    # If nothing fits the (conservative) budget, fall through to the SMALLEST
    # configuration rather than the default one: a shard pins slice_hidden_tiles,
    # so that is the only remaining move, and the real L1 is larger than the
    # fallback budget assumes.
    ladder = (
        ((RM_IN_DEPTH, RM_OUT_DEPTH), (1, RM_OUT_DEPTH), (1, 1)) if is_row_major else ((RM_IN_DEPTH, RM_OUT_DEPTH),)
    )
    block_rows, rm_in_depth, rm_out_depth = 1, ladder[-1][0], ladder[-1][1]
    for in_depth, out_depth in ladder:
        # A divisor keeps every block the same size, which is what lets the
        # resident-shard CB stay exactly full at every block boundary (the
        # in-place rewrite of x needs get_write_ptr() == get_read_ptr()).
        chosen = next((b for b in range(shard_rows, 0, -1) if shard_rows % b == 0 and _fits(b, in_depth, out_depth)), 0)
        if chosen:
            block_rows, rm_in_depth, rm_out_depth = chosen, in_depth, out_depth
            break

    grid = device.compute_with_storage_grid_size()
    return {
        "grid": (grid.x, grid.y),
        "row_tiles": row_tiles,
        "hidden_tiles": hidden_tiles,
        "num_row_groups": num_row_groups,
        "num_hidden_slices": num_hidden_slices,
        "slice_hidden_tiles": slice_tiles,
        "block_rows": block_rows,
        "in_depth": 1,
        "rect_w": 0,
        "rect_h": 0,
        "rm_in_depth": rm_in_depth,
        "rm_out_depth": rm_out_depth,
        "sharded": True,
        "cores": cores,
        "shard_rows": shard_rows,
        "shard_tiles": shard_tiles,
        "shard_h": shard_h,
        "shard_w": shard_w,
    }


def _plan(device, input_tensor, *, has_gamma, bytes_):
    """Pick (num_row_groups, num_hidden_slices, slice_hidden_tiles, block_rows)."""
    if is_sharded(input_tensor):
        return _plan_sharded(device, input_tensor, has_gamma=has_gamma, bytes_=bytes_)
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    row_tiles, hidden_tiles, _w, _sticks = _tile_geometry(input_tensor)
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    budget = _l1_working_budget(device)

    # Hidden-granularity floor: never slice so thin that the combine dominates a
    # 1-tile payload (feature_spec.py:343-346 measured geometries land at 4-8).
    # `_fanin_slice_cap` bounds the same tradeoff from the other end — see
    # FANIN_BALANCE_K for the measured U-curve in `s`.
    slice_cap = min(
        hidden_tiles,
        max(1, _div_up(hidden_tiles, HIDDEN_TILES_PER_CORE_FLOOR)),
        _fanin_slice_cap(hidden_tiles),
        gx * gy,
    )

    best = None
    for s, rect_w, rect_h in _rect_candidates(gx, gy):
        if s > slice_cap:
            continue
        slice_tiles = _div_up(hidden_tiles, s)
        # Tightness: every slice must own at least one tile (no idle rect core).
        if _div_up(hidden_tiles, slice_tiles) != s:
            continue
        if _footprint_bytes(1, slice_tiles, s, is_row_major=is_row_major, has_gamma=has_gamma, bytes_=bytes_) > budget:
            continue
        groups_geom = (gx // rect_w) * (gy // rect_h)
        groups = min(groups_geom, max(1, row_tiles))
        cores = groups * s
        # Occupancy first; then FEWER slices (fewer combines); then the WIDEST
        # rectangle, so a hidden line lies along a grid ROW — a column line is
        # 2.91x slower when bandwidth-bound (noc_placement/report.md:20-37).
        score = (cores, -s, rect_w)
        if best is None or score > best[0]:
            best = (score, groups, s, slice_tiles, rect_w, rect_h)

    if best is None:
        raise RuntimeError(
            "rms_norm: no partition of the hidden axis fits L1 for shape "
            f"{list(input_tensor.shape)} (TwoPassStreaming regime required — lamped)"
        )

    _score, num_row_groups, num_hidden_slices, slice_tiles, rect_w, rect_h = best

    # ---- the block/depth ladder ------------------------------------------------
    # Two knobs trade against each other inside ONE L1 budget, which is why they
    # are solved together rather than one after the other:
    #
    #   block_rows  — coarser is fewer fixed costs (one init set, one combine
    #                 round-trip, one fill/drain per block).
    #   in_depth    — a SECOND input buffer, so the reader can fill block b+1
    #                 while compute works on block b.  Worth nothing unless the
    #                 block is small enough that a core owns at least `depth` of
    #                 them, hence the cap below: this is the design's overlap lamp
    #                 in its stated form, "smaller block_rows + a second buffer",
    #                 never "same block, deeper CB".
    #
    # The ladder prefers the deeper buffer and degrades to the coarsest single
    # block when L1 cannot afford it — the same shape as `_plan_sharded`'s
    # rm-depth ladder.  ROW_MAJOR never takes a rung above 1: there the reader
    # fills `cb_rm_stage_in` (which has its OWN depth knob) and `cb_input_tiles`
    # is tilize's compute-side target, so a second one would overlap nothing.
    base, rem = divmod(row_tiles, num_row_groups)
    max_rows_full = base + (1 if rem else 0)
    budget = _l1_working_budget(device)

    def _fits_depth(b, depth):
        return (
            _footprint_bytes(
                b,
                slice_tiles,
                num_hidden_slices,
                is_row_major=is_row_major,
                has_gamma=has_gamma,
                bytes_=bytes_,
                in_depth=depth,
            )
            <= budget
        )

    ladder = (1,) if is_row_major else tuple(range(IN_CB_DEPTH, 0, -1))
    block_rows, in_depth = 1, 1  # nothing fits => the smallest configuration
    for depth in ladder:
        # A core that owns fewer than `depth` blocks has no block b+1 to prefetch,
        # so the extra buffer would be pure L1 with no overlap.  This is what keeps
        # the whole DECODE regime (one tile-row per core => max_rows_full == 1)
        # byte-identical to Refinement 3.
        if depth > 1 and max_rows_full < depth:
            continue
        cap = max(1, _div_up(max_rows_full, depth)) if depth > 1 else max_rows_full
        chosen = next((b for b in range(cap, 0, -1) if _fits_depth(b, depth)), 0)
        # ...and the block that survives the cap must still be worth prefetching.
        # Splitting a block in two buys one hidden read and COSTS one extra set of
        # per-block fixed costs (LLK init + format reconfig, a pipeline fill/drain,
        # and — when s > 1 — a whole gather + mcast + semaphore round trip).  A
        # block shorter than one in-flight NoC burst has no read worth hiding, so
        # the trade goes the wrong way: measured +1.4% on (1,1,2048,1024), whose
        # depth-2 block is 16 tiles.  DM_CHUNK_TILES *is* that burst, so the
        # threshold is that knob rather than a second literal beside it.
        if depth > 1 and chosen * slice_tiles < DM_CHUNK_TILES:
            chosen = 0
        if chosen:
            block_rows, in_depth = chosen, depth
            break

    return {
        "grid": (gx, gy),
        "row_tiles": row_tiles,
        "hidden_tiles": hidden_tiles,
        "num_row_groups": num_row_groups,
        "num_hidden_slices": num_hidden_slices,
        "slice_hidden_tiles": slice_tiles,
        "block_rows": block_rows,
        "in_depth": in_depth,
        "rect_w": rect_w,
        "rect_h": rect_h,
        "rm_in_depth": RM_IN_DEPTH,
        "rm_out_depth": RM_OUT_DEPTH,
        "sharded": False,
        "cores": None,
        "shard_rows": 0,
        "shard_tiles": 0,
        "shard_h": 0,
        "shard_w": 0,
    }


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def create_program_descriptor(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    *,
    gamma=None,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
) -> "ttnn.ProgramDescriptor":
    device = input_tensor.device()
    has_gamma = gamma is not None
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    in_tile = ttnn.tile_size(input_tensor.dtype)
    out_tile = ttnn.tile_size(output_tensor.dtype)
    gamma_tile = ttnn.tile_size(gamma.dtype) if has_gamma else in_tile
    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    bytes_ = {
        "in_tile": in_tile,
        "out_tile": out_tile,
        "gamma_tile": gamma_tile,
        "stat_tile": stat_tile,
        "bf16_tile": bf16_tile,
    }

    plan = _plan(device, input_tensor, has_gamma=has_gamma, bytes_=bytes_)
    gx, gy = plan["grid"]
    Rt = plan["row_tiles"]
    Wt = plan["hidden_tiles"]
    G = plan["num_row_groups"]
    S_COUNT = plan["num_hidden_slices"]
    S = plan["slice_hidden_tiles"]
    B = plan["block_rows"]
    in_depth = plan["in_depth"]
    rect_w, rect_h = plan["rect_w"], plan["rect_h"]
    sharded = plan["sharded"]
    shard_rows = plan["shard_rows"]
    shard_tiles = plan["shard_tiles"]
    shard_w = plan["shard_w"]
    rm_in_depth = plan["rm_in_depth"]
    rm_out_depth = plan["rm_out_depth"]

    shape = list(input_tensor.shape)
    W = shape[-1]
    in_elem = _elem_bytes(input_tensor)
    out_elem = _elem_bytes(output_tensor)
    gamma_elem = _elem_bytes(gamma) if has_gamma else in_elem
    total_sticks = input_tensor.buffer_num_pages() if is_row_major else 0

    mask_valid_w = W % TILE_DIM
    mask_enabled = (mask_valid_w != 0) and not is_row_major

    # A TILE-layout gamma is a [W] vector padded to a whole tile-row: only ROW 0
    # of each of its Wt tiles carries data, and `BroadcastDim::Row` is the only
    # consumer, so the reader fetches the two row-0 face segments (2 x 32 elems)
    # instead of the whole tile.  That is 32x fewer gamma bytes off DRAM — in the
    # decode regime (Rt == 1) gamma is otherwise a full third of the op's traffic.
    #
    # Refused for block-float gamma: a bfp8 tile's faces share an exponent header,
    # so a row-slice of the page is not a decodable tile.
    _BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b,)
    gamma_row0_only = has_gamma and gamma.layout == ttnn.TILE_LAYOUT and gamma.dtype not in _BLOCK_FLOAT_DTYPES

    # ---- core assignment ----
    # Interleaved: G rectangles of (rect_w x rect_h) cores, chosen by _plan.
    # Sharded:     G groups of S_COUNT consecutive cores of the SHARD grid, in
    #              shard order (group r == the r-th height-shard, slice c == the
    #              c-th width-shard) — the same 2D partition, read off the spec.
    groups = []  # list of dicts: cores (slice order), rows, sticks
    if sharded:
        core_list = plan["cores"]
        shard_h = plan["shard_h"]
        for r in range(G):
            gcores = core_list[r * S_COUNT : (r + 1) * S_COUNT]
            if is_row_major:
                sticks = max(0, min(shard_h, total_sticks - r * shard_h))
                rows = _div_up(sticks, TILE_DIM)
            else:
                sticks = 0
                rows = max(0, min(shard_rows, Rt - r * shard_rows))
            groups.append(
                {
                    "origin": gcores[0],
                    "cores": gcores,
                    "row_start": 0,  # shard-local: the shard IS this group's rows
                    "core_row_tiles": rows,
                    "local_sticks": sticks,
                    "num_blocks": _div_up(rows, B) if rows > 0 else 0,
                }
            )
    else:
        rects_per_grid_row = gx // rect_w
        row_base, row_rem = divmod(Rt, G)
        row_cursor = 0
        for r in range(G):
            ox = (r % rects_per_grid_row) * rect_w
            oy = (r // rects_per_grid_row) * rect_h
            cores = []
            for dy in range(rect_h):
                for dx in range(rect_w):
                    cores.append((ox + dx, oy + dy))
            rows = row_base + (1 if r < row_rem else 0)
            groups.append(
                {
                    "origin": (ox, oy),
                    "cores": cores,
                    "row_start": row_cursor,
                    "core_row_tiles": rows,
                    "local_sticks": total_sticks,
                    "num_blocks": _div_up(rows, B) if rows > 0 else 0,
                }
            )
            row_cursor += rows

    active_groups = [g for g in groups if g["core_row_tiles"] > 0]

    def _bbox(cores):
        xs = [c[0] for c in cores]
        ys = [c[1] for c in cores]
        return ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))

    # Kernels run on the cores that actually own data.  CBs and semaphores are
    # declared on the union of the row-group BOUNDING BOXES, which is the same
    # set for every rectangular group (interleaved, HEIGHT, BLOCK) and a superset
    # for a ragged WIDTH shard grid ("N full grid rows + a partial row").
    # `Mcast2D` takes the bounding box as THE rect, so the broadcast lands on the
    # few non-member cores too; declaring the CB there reserves that L1 (and the
    # divergent ack count below keeps the handshake counting only real members),
    # which is what makes a non-rectangular shard grid safe instead of refused.
    if sharded:
        kernel_cores_crs = input_tensor.memory_config().shard_spec.grid
        # With no broadcast there is no bounding box to cover, so the CBs sit on
        # exactly the shard grid.
        cb_cores_crs = (
            kernel_cores_crs if S_COUNT == 1 else ttnn.CoreRangeSet([_bbox(g["cores"]) for g in active_groups])
        )
    else:
        kernel_cores_crs = ttnn.CoreRangeSet([_bbox(g["cores"]) for g in active_groups])
        cb_cores_crs = kernel_cores_crs

    # ---- mcast wire: one Mcast2D per row-group rect (identical CT for all) ----
    #
    # `handshake` (the receiver->sender readiness ack) is a PERF KNOB, not a
    # constant: it costs the root s-1 inbound remote atomics and a wait, per
    # block, and it buys exactly one thing — the guarantee that broadcast n+1
    # does not overwrite a landing buffer still holding broadcast n.
    #
    # With ONE broadcast per kernel (num_blocks == 1: the whole decode regime,
    # where the row-group is a single tile-row) there IS no broadcast n+1, and
    # cb_rms_recip is untouched at boot, so the ack is pure cost.  Dropping it is
    # safe because the remaining ordering edge still holds: every receiver
    # constructs its ReceiverPipe (which inits its own data_ready flag to
    # INVALID) at kernel boot, and the root cannot send until it has gathered all
    # s partials -- which needs every contributor's compute+writer to have run,
    # i.e. strictly after that core's reader passed the ctor.  So the flag can
    # never be clobbered by a signal that arrives before the ctor.
    #
    # As soon as a row-group has more than one block the ack is load-bearing and
    # comes back on.
    single_shot_bcast = all(g["num_blocks"] <= 1 for g in active_groups)
    mcast_by_group = {}
    if S_COUNT > 1:
        cfg = ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=not single_shot_bcast,
            sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
        )
        for idx, g in enumerate(active_groups):
            ox, oy = g["origin"]
            rect_crs = ttnn.CoreRangeSet([_bbox(g["cores"])])
            # num_active = the REAL receiver count.  Equals the dense fan-out for
            # a rectangular group; smaller when the bounding box holds cores that
            # receive the broadcast but never ack (ragged WIDTH shard grid).
            mcast_by_group[idx] = ttnn.Mcast2D(device, rect_crs, ttnn.CoreCoord(ox, oy), cfg, S_COUNT - 1)
        mcast_ct = list(mcast_by_group[0].compile_time_args())
    else:
        mcast_ct = [0, 0, 0, 0, 0]
    assert len(mcast_ct) == 5

    # ---- circular buffers (all on the SAME core set so the L1 map is identical
    #      across cores; the cross-core gather and the mcast landing both rely on
    #      a peer's CB address being derivable from the local one) ----
    def _cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=cb_cores_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        _cb(CB_SQ_PARTIALS, B, stat_tile, ttnn.float32),
        _cb(CB_RMS_RECIP, B, stat_tile, ttnn.float32),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16),
        _cb(CB_THREAD_SYNC, 1, bf16_tile, ttnn.bfloat16),
    ]

    # x / out placement.  A physical shard is consumed NATIVELY: the CB is bound
    # to the caller's L1 buffer (zero-copy), never re-read through a
    # TensorAccessor.  On TILE the shard IS cb_input_tiles / cb_output_tiles, so
    # there is no copy at all; on ROW_MAJOR the shard is bound to its own CB and
    # the (already mandatory) tilize staging reads it core-LOCALLY.
    if sharded:
        if is_row_major:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SHARD_IN, input_tensor))
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_SHARD_OUT, output_tensor))
            cbs.append(_cb(CB_INPUT_TILES, B * S, in_tile, input_tensor.dtype))
        else:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor))
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor))
    else:
        # capacity == live set exactly: the in-place rewrite of x needs
        # get_write_ptr == get_read_ptr, which only holds at a full CB.
        cbs.append(_cb(CB_INPUT_TILES, in_depth * B * S, in_tile, input_tensor.dtype))
        if not is_row_major:
            cbs.append(_cb(CB_OUTPUT_TILES, OUT_CB_DEPTH * S, out_tile, output_tensor.dtype))

    if mask_enabled:
        # Only the W-mask path (TILE layout with W % 32 != 0) touches this CB;
        # both the reader's prepare_reduce_mask and the compute kernel's
        # mask_tail_block are gated on the same predicate.  Matches l1_ledger.md.
        cbs.append(_cb(CB_W_MASK, 1, bf16_tile, ttnn.bfloat16))
    if has_gamma:
        # A ROW_MAJOR *shard*'s slice can start off the 64 B DRAM boundary (its
        # width granule is the L1 alignment, not the tile), so the reader reads
        # one DRAM-aligned burst covering the whole slice into scratch pages past
        # the S real tiles and hand-places the row-0 lanes.  Sized to hold the
        # slice plus the alignment slack.
        gamma_scratch = (
            _div_up(DRAM_ALIGN_BYTES + S * TILE_DIM * gamma_elem, gamma_tile) if (sharded and is_row_major) else 0
        )
        cbs.append(_cb(CB_GAMMA_TILES, S + gamma_scratch, gamma_tile, gamma.dtype))
    if S_COUNT > 1:
        cbs.append(_cb(CB_GATHERED_PARTIALS, S_COUNT * B, stat_tile, ttnn.float32))
        cbs.append(_cb(CB_RMS_BCAST, B, stat_tile, ttnn.float32))
    if is_row_major:
        cbs.append(_cb(CB_RM_STAGE_IN, rm_in_depth * S, in_tile, input_tensor.dtype))
        cbs.append(_cb(CB_RM_STAGE_OUT, rm_out_depth * S, out_tile, output_tensor.dtype))

    # ---- compile-time args ----
    # Two DIFFERENT numbers, and conflating them is what pinned IN_CB_DEPTH at 1:
    #
    #   in_wait_tiles     — how many pages compute holds at once (its wait count).
    #   in_capacity_tiles — cb_input_tiles' whole capacity, which is what the
    #                       in-place pack index is taken MODULO, because a pack
    #                       lands at `CB base + index*page` and the read window
    #                       walks the capacity in BLOCK_TILES steps.
    #
    # They coincide on every path except (a) TILE + sharded, where the CB IS the
    # caller's resident shard and compute waits the full shard window so the
    # reader can keep it exactly full at every block boundary, and (b) a
    # double-buffered input, where capacity is IN_CB_DEPTH blocks but compute
    # still waits exactly one.  Capacity is always a whole multiple of BLOCK_TILES
    # (a shard's block_rows is a divisor of shard_rows), so the read pointer never
    # wraps mid-block and `% in_capacity_tiles` is exact.
    in_wait_tiles = shard_tiles if (sharded and not is_row_major) else (B * S)
    in_capacity_tiles = shard_tiles if (sharded and not is_row_major) else (in_depth * B * S)
    in_shard_page = input_tensor.buffer_aligned_page_size() if sharded else 0
    out_shard_page = output_tensor.buffer_aligned_page_size() if sharded else 0

    reader_ct = list(mcast_ct) + [
        S,
        B,
        S_COUNT,
        1 if has_gamma else 0,
        1 if is_row_major else 0,
        1 if (has_gamma and gamma.layout == ttnn.TILE_LAYOUT) else 0,
        Wt,
        in_tile,
        gamma_tile,
        in_elem,
        gamma_elem,
        SEM_GATHER_PROGRESS,
        stat_tile,
        DM_CHUNK_TILES,
        rm_in_depth * S,
        1 if mask_enabled else 0,
        1 if sharded else 0,
        in_wait_tiles,
        in_shard_page,
        1 if gamma_row0_only else 0,
        in_capacity_tiles,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = [
        S,
        B,
        S_COUNT,
        1 if is_row_major else 0,
        out_tile,
        stat_tile,
        SEM_GATHER_PROGRESS,
        Wt,
        out_elem,
        DM_CHUNK_TILES,
        1 if sharded else 0,
        out_shard_page,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # The gamma fusion rung (see GAMMA_FUSE_MIN_ROW_TILES): worth its one-off gamma
    # expansion only where a core owns more than one tile-row.  `max_core_row_tiles`
    # is the deepest per-core row count in this program, so the whole grid runs one
    # kernel variant (a per-core CT arg does not exist).
    max_core_row_tiles = max((g["core_row_tiles"] for g in active_groups), default=0)
    gamma_fused = has_gamma and GAMMA_FUSE_MIN_ROW_TILES > 0 and max_core_row_tiles >= GAMMA_FUSE_MIN_ROW_TILES

    dest_block_tiles = DEST_BLOCK_TILES if max_core_row_tiles >= DEST_BLOCK_MIN_ROW_TILES else 1

    compute_ct = [
        S,
        B,
        S_COUNT,
        1 if has_gamma else 0,
        1 if is_row_major else 0,
        1 if mask_enabled else 0,
        in_wait_tiles,
        in_capacity_tiles,
        1 if gamma_fused else 0,
        dest_block_tiles,
    ]

    # ---- per-core runtime args ----
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inv_w_bits = _f32_bits(1.0 / float(W))
    eps_bits = _f32_bits(epsilon)
    input_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    output_addr = output_tensor.buffer_address()

    for idx, g in enumerate(active_groups):
        ox, oy = g["origin"]
        root_virt = device.worker_core_from_logical_core(ttnn.CoreCoord(ox, oy))
        mc = mcast_by_group.get(idx)
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            slice_base = slice_index * S  # this core's hidden slice, in TILES
            # ...and in ELEMENTS.  The two agree (slice_base*32) everywhere except
            # a ROW_MAJOR shard, whose width granule is the L1 alignment (8 for
            # bf16), not 32 — so the slice boundary need not be tile-aligned.
            slice_elem_base = slice_index * shard_w if (sharded and is_row_major) else slice_base * TILE_DIM
            slice_elems = shard_w if (sharded and is_row_major) else S * TILE_DIM
            valid_w = max(0, min(slice_elems, W - slice_elem_base))
            valid_tiles = _div_up(valid_w, TILE_DIM)
            is_root = 1 if (cx, cy) == (ox, oy) else 0
            owns_tail = slice_index == (S_COUNT - 1)
            mask_local_col = (Wt - 1) - slice_base if (mask_enabled and owns_tail) else 0xFFFFFFFF

            mcast_rt = list(mc.runtime_args(ttnn.CoreCoord(cx, cy))) if mc is not None else [0, 0, 0, 0]

            reader_rt[cx][cy] = list(mcast_rt) + [
                input_addr,
                gamma_addr,
                g["row_start"],
                g["core_row_tiles"],
                g["num_blocks"],
                slice_base,
                valid_tiles,
                valid_w,
                is_root,
                mask_valid_w if (mask_enabled and owns_tail) else 0,
                g["local_sticks"],
                slice_elem_base,
            ]

            writer_rt[cx][cy] = [
                output_addr,
                g["row_start"],
                g["core_row_tiles"],
                g["num_blocks"],
                slice_base,
                valid_tiles,
                valid_w,
                root_virt.x,
                root_virt.y,
                slice_index,
                g["local_sticks"],
                slice_elem_base,
            ]

            compute_rt[cx][cy] = [
                g["num_blocks"],
                is_root,
                mask_local_col,
                inv_w_bits,
                eps_bits,
            ]

    # ---- kernels ----
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=kernel_cores_crs,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=kernel_cores_crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=kernel_cores_crs,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=cb_cores_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=cb_cores_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_GATHER_PROGRESS, core_ranges=cb_cores_crs, initial_value=0),
    ]

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
