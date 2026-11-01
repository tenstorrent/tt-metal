# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: geometry, select_blocking(), CBs, kernels, per-core args.

Blocking (op_design.md -> Blocking Model):
  * row axis (tile-rows, leading dims folded): split over row-groups across the whole Tensix grid;
    each group's assignment is processed in blocks of `block_row_tiles` tile-rows.
  * W axis: split over `w_split_cores` (k) consecutive Tensix cores (one grid row while k <= grid_x, else
    row-major across grid rows); every member keeps its W share resident (cb_x) across pass 1 and pass 2,
    and the k members combine their per-row sum-of-squares partials over the NoC (k > 1 only): symmetric
    all-gather for k <= 8, per-row owner gather + inv_rms return beyond (Refinement 5).
Every knob below (k, block_row_tiles, x_block_depth, rm_chunk_w_tiles, out_unit_tiles) is computed
once here and handed to the kernels as a CT/RT arg; the kernels derive all trip counts from them.

HEIGHT / WIDTH / BLOCK_SHARDED (Refinements 2, 4): the shard spec fixes the work split — one group per shard row
(BLOCK / HEIGHT) or a single group (WIDTH), k = shard columns (HEIGHT: k = 1, no combine), members and their NoC coordinates from the shard
grid (orientation-aware, groups may span grid rows). Block = the resident shard (sub-blocked along rows
only under L1 pressure). TILE: cb_x / cb_out are backed on the input / output shards (zero-copy). RM:
the resident sticks are copied locally into the tilize staging CB (never gathered across Tensix cores).
Combine topology: k <= K_SYM_MAX_* symmetric unicast all-gather; beyond, a per-row owner gathers the
k partials, combines + finalizes, and returns inv_rms to every member (L1 per member ~ k * ceil(b/k)).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# CB indices — must match kernels/rms_norm_common.hpp.
CB_X_RM = 0  # RM: raw stick pieces (tile-sized pages) -> tilize
CB_X = 1  # resident x block (ping)
CB_GAMMA = 2  # expanded gamma slice, program lifetime
CB_SUMSQ = 3  # per tile-row DEST-accumulated sum of x^2 (pass-1 output)
CB_PARTIAL = 4  # k > 1: this member's per-row partial (column 0)
CB_GATHERED = 5  # k > 1: k partials per row, all-gathered over the NoC
CB_INV_RMS = 6  # per tile-row rsqrt(mean + eps) (column 0)
CB_OUT = 7  # normalized tiles (TILE: -> writer; RM: -> untilize)
CB_OUT_RM = 8  # RM: untilized stick pieces -> writer
CB_X_ALT = 9  # resident x block (pong) — only when x_block_depth == 2
CB_SUMSQ_ACC = 10  # sharded TILE W tail: raw sum-of-squares of the full W tiles (reduce Accumulate CB)
CB_TAIL_MASK = 11  # sharded TILE W tail: 0/1 row mask of the valid columns of the last W tile
CB_INV_OWNED = 12  # owner combine (k > K_SYM_MAX_*): inv_rms of the tile-rows this member owns

SEM_PARTIALS_ARRIVED = 0
SEM_INV_ARRIVED = 1  # owner combine: inv_rms tiles returned by the row owners
SEM_GAMMA_MCAST_FIRST = 2  # gamma multicast channel (ttnn.Mcast allocates its semaphores from this id)
WRITER_NOC_XY_ARG_BASE = 13  # writer RT index of the first peer (x, y) pair — passed to the kernel as a CT arg

# Knob constants (design Phase 0 values).
# Interleaved W split cap: the writer carries one (x, y) NoC pair per member as runtime args, so k is bounded by
# the per-kernel runtime-arg limit (tt-metalium kernel_types.hpp max_runtime_args = 341). Beyond K_SYM_MAX_INTERLEAVED
# the owner combine runs; beyond grid_x a group spans grid rows (members placed row-major over the grid).
MAX_RUNTIME_ARGS = 341
K_CAP_MAX = (MAX_RUNTIME_ARGS - WRITER_NOC_XY_ARG_BASE) // 2
# Combine topology: symmetric all-gather up to K_SYM_MAX_* members, per-row owner gather + inv_rms return beyond.
# Interleaved (W split k <= 8): catalog tensix_all_reduce (k <= 8 unicast all-gather). Sharded: the owner
# topology for every k > 1 — measured BH p150b (bf16, HiFi2, 16-bit DEST): BLOCK (7168,1024) [736,128] 8x10
# 70.8 -> 32.3 us, (8192,1024) [1024,128] 8x8 96.7 -> 42.7 us (each member combines + finalizes rows/k
# tile-rows instead of all of them); 1- and 2-tile-row shards flat (<= 3%).
K_SYM_MAX_INTERLEAVED = 8
K_SYM_MAX_SHARDED = 1
# Interleaved: the Tensix cores sharing a W slice get gamma by one multicast (one DRAM read per slice) instead
# of each reading it (the writer loads it, off the reader's x stream). BH p150b, perf config: the per-core load
# was a 131 us serial prefix of (8192,7168) (130 Tensix cores reading the same 224 gamma tiles, 2 x 64 B reads
# each); measured 771 -> 630 us (8192,7168), 90.8 -> 76.3 us (7168,1024), (640,6144) k=6 51.9 -> 45.7 us.
GAMMA_MCAST = True
# Sharded k = 1 (HEIGHT, or a single shard column): every Tensix core holds the whole gamma, so the same one-group
# multicast applies (one DRAM read of gamma per launch instead of one per Tensix core). WIDTH / BLOCK k > 1 keep the
# per-core slice load (a shard grid column need not hold a single W slice).
GAMMA_MCAST_SHARDED_K1 = True
# k > 1, per-core gamma load (no multicast): the writer issues the gamma DRAM reads at kernel start but finishes the
# load (barrier + expand) only after block 0's partials are sent / its owner duties are done, so the cross-core
# combine no longer waits on it. Pays when pass 1 is short (few tile-rows per block): BH p150b, perf config,
# blocks of 1-2 tile-rows — (1,1,32,32768) k=49 24.3 -> 20.8 us, (160,11008) k=25 29.4 -> 24.8 us, (352,4096)
# 25.9 -> 24.2 us, WIDTH (32,7168) 9.3 -> 8.3 us. With long blocks the early load is already hidden behind pass 1
# and the late expand only delays pass 2: BLOCK (7168,1024) [736,128] (23 tile-rows) 29.9 -> 30.8 us. Gated on
# block_row_tiles <= GAMMA_AFTER_SEND_MAX_ROWS.
GAMMA_AFTER_SEND = True
GAMMA_AFTER_SEND_MAX_ROWS = 8
# Combine signal fence: flush (writes left this Tensix core) instead of a write barrier before each semaphore
# increment of the partial exchange / inv_rms return (catalog tensix_all_reduce, design perf lamp "Signal fence").
SIGNAL_AFTER_FLUSH = True
PARTIAL_FACES_ONLY = True  # ship faces 0/2 of a partial / inv_rms tile (column 0 lives there), not the whole tile
OUT_DEPTH = 2  # cb_out / cb_out_rm buffer depth
# Streamed gamma (fallback when the resident expanded gamma slice does not fit L1 — e.g. HEIGHT k = 1 with a wide W
# beside the two resident shards): gamma flows through a GAMMA_STREAM_DEPTH-chunk ring of <= GAMMA_STREAM_CHUNK_CAP
# tiles, re-loaded per (block, chunk). The largest chunk that fits is chosen.
GAMMA_STREAM_CHUNK_CAP = 16
GAMMA_STREAM_DEPTH = 2
GAMMA_STREAM_FORCE_CHUNK = 0  # > 0: stream gamma in chunks of this many tiles even when resident fits (test hook)
RM_IN_DEPTH = 2  # cb_x_rm buffer depth
OUT_UNIT_CAP = 16  # TILE writer batch (tiles per wait/flush/pop)
RM_CHUNK_CAP = 32  # balanced RM streaming chunk cap (tiles)
RM_PIECE_FLOOR_BYTES = 512  # occupancy split must keep RM stick pieces >= this
OCC_TILES_PER_MEMBER = 4  # occupancy split keeps >= 4 W tiles per member
COMBINE_TILE_COST = 3  # k > 1: per-tile-row collapse + exchange + combine cost, in x-tile equivalents
OWNER_MEMBER_COST = 0.4  # owner combine: per gathered partial on the owner (add + return write), x-tile equivalents
# Whole assignment fits: split it into this many depth-2 blocks (1 = one block per core). Measured 2 on BH p150b
# (perf config): (8192,7168) 630 -> 673 us, (7168,1024) 76 -> 81 us — the k = 1 prefill shapes are DRAM-write-
# bound (compute stubbed: same time; write-only 438 vs read-only 313 us at 8192x7168), so 1 stays the default.
OVERLAP_BLOCKS = 1


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class Geometry:
    W: int
    H: int
    num_images: int
    image_row_tiles: int
    tensor_row_tiles: int
    tensor_w_tiles: int
    w_valid_in_last_tile: int


def derive_geometry(shape) -> Geometry:
    shape = [int(d) for d in shape]
    W, H = shape[-1], shape[-2]
    num_images = 1
    for d in shape[:-2]:
        num_images *= d
    image_row_tiles = _div_up(H, TILE)
    tensor_w_tiles = _div_up(W, TILE)
    return Geometry(
        W=W,
        H=H,
        num_images=num_images,
        image_row_tiles=image_row_tiles,
        tensor_row_tiles=num_images * image_row_tiles,
        tensor_w_tiles=tensor_w_tiles,
        w_valid_in_last_tile=W - TILE * (tensor_w_tiles - 1),
    )


def acc_dtype(fp32_dest_acc_en: bool):
    """Format of every DEST-written intermediate CB (sumsq / partial / gathered / inv_rms).

    Follows the DEST width: with 16-bit DEST the value is already rounded before pack_tile(), so a Float32
    page would carry no extra bits while doubling packer/unpacker/NoC bytes and L1.
    """
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


class _Sizes:
    def __init__(self, x_dtype, gamma_dtype, is_rm, has_gamma, fp32_dest_acc_en):
        self.T_in = ttnn.tile_size(x_dtype)
        self.T_out = self.T_in
        self.T_g = ttnn.tile_size(gamma_dtype) if has_gamma else 0
        self.T_acc = ttnn.tile_size(acc_dtype(fp32_dest_acc_en))
        self.is_rm = is_rm
        self.has_gamma = has_gamma


def _normalize_k(tw: int, k: int):
    s = _div_up(tw, k)
    return _div_up(tw, s), s


def _out_unit(slice_w_tiles: int) -> int:
    """TILE writer batch (tiles per wait/write/flush/pop) — also the cb_out depth unit."""
    return min(slice_w_tiles, OUT_UNIT_CAP)


def _rm_chunk(slice_w_tiles: int, cap: int = RM_CHUNK_CAP):
    chunks = _div_up(slice_w_tiles, cap)
    c = _div_up(slice_w_tiles, chunks)
    return c, chunks * c


@dataclass(frozen=True)
class PlanOpts:
    """Regime switches that change the CB inventory (one source of truth for footprint() and the CB list)."""

    shard_backed: bool = False  # TILE sharded: cb_x / cb_out alias the shards (tensor buffers, not CB budget)
    owner_combine: bool = False  # k > K_SYM_MAX_*: per-row owner gather + return instead of the all-gather
    gather_depth: int = 2  # symmetric all-gather: blocks of cb_gathered (1 when every core runs one block)
    mask_tail: bool = False  # sharded TILE, W % 32 != 0: tail tile's x^2 masked inside the collapse reduce
    gamma_chunk: int = 0  # 0: gamma slice resident (program lifetime); G > 0: streamed in G-tile chunks (ring)
    rm_chunk_cap: int = RM_CHUNK_CAP  # RM: cap of the streaming chunk C (lowered only when the plan does not fit)


def cb_plan(sz: _Sizes, k: int, slice_w: int, b: int, d: int, opts: PlanOpts = PlanOpts()) -> dict:
    """Single source of truth for every CB's capacity: {cb_index: (pages, page_bytes)}.

    Both footprint() (the L1-fit solve in select_blocking) and create_program_descriptor() (the CB
    declarations) read this dict, so a knob change lands in exactly one place (l1_ledger.md rows).
    """
    if sz.is_rm:
        c, pitch = _rm_chunk(slice_w, opts.rm_chunk_cap)
    else:
        c, pitch = 1, slice_w
    plan = {
        CB_SUMSQ: (b, sz.T_acc),
        CB_INV_RMS: (b, sz.T_acc),
    }
    if not opts.shard_backed:
        plan[CB_X] = (b * pitch, sz.T_in)  # resident x block
        if d == 2:
            plan[CB_X_ALT] = (b * pitch, sz.T_in)  # second half of the depth-2 x budget
    if sz.has_gamma:
        plan[CB_GAMMA] = (GAMMA_STREAM_DEPTH * opts.gamma_chunk if opts.gamma_chunk else pitch, sz.T_g)
    if k > 1:
        plan[CB_PARTIAL] = (b, sz.T_acc)
        if opts.owner_combine:
            owned = _div_up(b, k)  # tile-rows a member owns per block (row r -> member r % k)
            plan[CB_GATHERED] = (owned * k, sz.T_acc)  # k partials per owned row (uniform -> lockstep slots)
            plan[CB_INV_OWNED] = (owned, sz.T_acc)
        else:
            plan[CB_GATHERED] = (opts.gather_depth * k * b, sz.T_acc)  # lockstep slot-reuse mechanism cap
    if opts.mask_tail:
        plan[CB_SUMSQ_ACC] = (b, sz.T_acc)
        plan[CB_TAIL_MASK] = (1, sz.T_acc)
    if sz.is_rm:
        plan[CB_X_RM] = (RM_IN_DEPTH * c, sz.T_in)
        plan[CB_OUT] = (b * c, sz.T_out)  # spans one chunk column of the block (untilize input)
        plan[CB_OUT_RM] = (OUT_DEPTH * c, sz.T_out)
    elif not opts.shard_backed:
        plan[CB_OUT] = (OUT_DEPTH * _out_unit(slice_w), sz.T_out)
    return plan


def footprint(sz: _Sizes, tw: int, k: int, b: int, d: int, opts: PlanOpts = PlanOpts()) -> int:
    """Per-Tensix-core CB bytes — the l1_ledger.md expression, summed from cb_plan()."""
    k, s = _normalize_k(tw, k)
    return _plan_bytes(cb_plan(sz, k, s, b, d, opts))


def _plan_bytes(plan: dict) -> int:
    return sum(pages * page_bytes for pages, page_bytes in plan.values())


@dataclass
class Member:
    """One Tensix core's share of the work: a run of tile-rows x a W slice."""

    core: object  # logical ttnn.CoreCoord
    m: int  # member index inside its group (W-slice order)
    row_start: int  # first global tile-row (interleaved addressing; sharded: informational)
    core_rows: int  # tile-rows on this core
    w_tile_start: int  # first global W tile
    core_w: int  # W tiles this core reduces (valid, not padding)
    col0: int  # first global column (datums)
    valid_cols: int  # valid columns of this slice (datums)
    valid_sticks: int = 0  # RM sharded: valid sticks (rows) in the resident shard


@dataclass
class Blocking:
    k: int  # w_split_cores (members per group)
    slice_w_tiles: int  # W tiles of every non-last member
    rm_chunk_w_tiles: int  # C (RM streaming quantum along W; TILE: unused, = 1)
    x_row_pitch: int  # tile pitch of one tile-row inside cb_x (uniform across cores)
    groups: list  # per group: list of k Member
    block_row_tiles: int
    x_block_depth: int
    out_unit_tiles: int
    opts: PlanOpts
    footprint: int


def _pitch(sz: _Sizes, slice_w: int, rm_chunk_cap: int = RM_CHUNK_CAP):
    """(rm_chunk_w_tiles, x_row_pitch) for a slice of `slice_w` tiles."""
    return _rm_chunk(slice_w, rm_chunk_cap) if sz.is_rm else (1, slice_w)


def _solve_rows(
    sz: _Sizes,
    k: int,
    slice_w: int,
    crt_max: int,
    budget: int,
    opts: PlanOpts,
    allow_depth2: bool,
    overlap_blocks: int = 1,
):
    """Largest block_row_tiles (and x_block_depth) whose CB plan fits `budget`.

    When the whole assignment fits: overlap_blocks > 1 (and depth 2 allowed) splits it into ceil(crt_max /
    overlap_blocks)-row blocks at depth 2, so the read of block b+1 overlaps compute + write of block b
    (design perf lamp "Overlap (one-block assignment)"); overlap_blocks == 1 keeps one block per core.
    Otherwise the largest sub-block at depth 2 (TILE interleaved) / 1. The symmetric all-gather needs a
    2-block cb_gathered only when a core runs > 1 block.
    """

    def fits(b, d):
        single = b >= crt_max
        o = replace(opts, gather_depth=1 if single else 2)
        return _plan_bytes(cb_plan(sz, k, slice_w, b, d, o)) <= budget, o

    if allow_depth2 and overlap_blocks > 1 and crt_max > 1:
        b_ov = _div_up(crt_max, min(overlap_blocks, crt_max))
        ok, o = fits(b_ov, 2)
        if ok:
            return b_ov, 2, o
    ok, o = fits(crt_max, 1)
    if ok:
        return crt_max, 1, o
    d = 2 if allow_depth2 else 1
    for cand in range(crt_max, 0, -1):
        ok, o = fits(cand, d)
        if ok:
            return cand, (1 if cand >= crt_max else d), o
    ok, o = fits(1, 1)
    if not ok:
        raise RuntimeError(f"rms_norm: one tile-row block does not fit L1 ({budget} B per core)")
    return 1, 1, o


def _sharded_plan_candidates(sz: _Sizes, slice_w: int, opts: PlanOpts):
    """Sharded CB regimes in preference order; the first whose one-tile-row plan fits L1 is used.

    Default: resident gamma slice (loaded / multicast once) and the RM chunk at RM_CHUNK_CAP. Wide W beside two
    resident shards may not fit that (HEIGHT k = 1 holds the whole row), so fall back to
      TILE: gamma streamed in the largest G <= GAMMA_STREAM_CHUNK_CAP chunks (per block, 2-chunk ring);
      RM  : per halved chunk cap C (staging / cb_out / untilize buffers scale with C): resident gamma, then gamma
            streamed with G = C (pass 2 already walks C-tile windows). The tilized x row stays resident.
    GAMMA_STREAM_FORCE_CHUNK (test hook) skips the resident-gamma candidates (TILE: also caps G).
    """
    force = bool(GAMMA_STREAM_FORCE_CHUNK) and sz.has_gamma
    if not sz.is_rm:
        if not force:
            yield opts
        if sz.has_gamma:
            for g in range(min(GAMMA_STREAM_FORCE_CHUNK or GAMMA_STREAM_CHUNK_CAP, slice_w), 0, -1):
                yield replace(opts, gamma_chunk=g)
        return
    cap = RM_CHUNK_CAP
    while True:
        o = replace(opts, rm_chunk_cap=cap)
        if not force:
            yield o
        if sz.has_gamma:
            yield replace(o, gamma_chunk=_rm_chunk(slice_w, cap)[0])
        if cap == 1:
            return
        cap //= 2


def _solve_rows_sharded(sz: _Sizes, k: int, slice_w: int, crt_max: int, budget: int, opts: PlanOpts):
    """_solve_rows over _sharded_plan_candidates(): the first regime in which one tile-row block fits."""
    for cand in _sharded_plan_candidates(sz, slice_w, opts):
        try:
            return _solve_rows(sz, k, slice_w, crt_max, budget, cand, allow_depth2=False)
        except RuntimeError:
            continue
    raise RuntimeError(
        f"rms_norm: one tile-row block does not fit L1 ({budget} B per core), even with streamed gamma / RM chunk 1"
    )


def _max_groups(k: int, grid_x: int, grid_y: int) -> int:
    """Row-groups a W split k fits on the grid: k <= grid_x keeps each group inside one grid row (grid_x // k per
    row — a grid column then holds a single W slice, which the gamma multicast relies on); k > grid_x packs the
    groups row-major over the whole grid (a group spans grid rows)."""
    return (grid_x // k) * grid_y if k <= grid_x else (grid_x * grid_y) // k


def _occ_cost(trt: int, tw: int, k: int, grid_x: int, grid_y: int) -> float:
    """Per-core critical-path work of a W split k, in x-tile equivalents.

    Rows of the busiest group x (its slice width + the per-tile-row cost of the k > 1 collapse / exchange /
    combine). The owner combine (k > K_SYM_MAX_INTERLEAVED) adds, on the busiest owner, ceil(rows / k) owned rows
    x k gathered partials (one combine add + one inv_rms return write each): with few rows it grows with k.
    """
    groups = min(_max_groups(k, grid_x, grid_y), trt)
    rows = _div_up(trt, groups)
    cost = rows * (_div_up(tw, k) + (COMBINE_TILE_COST if k > 1 else 0))
    if k > K_SYM_MAX_INTERLEAVED:
        cost += _div_up(rows, k) * k * OWNER_MEMBER_COST
    return cost


def _choose_k(
    geo: Geometry, sz: _Sizes, grid_x: int, grid_y: int, k_fit: int, k_cap: int, elem_bytes: int, budget: int
) -> int:
    """w_split_cores: the k >= k_fit that minimizes the busiest Tensix core's work (catalog width_split).

    A split is a candidate only while every member keeps >= OCC_TILES_PER_MEMBER W tiles (RM: stick pieces
    >= RM_PIECE_FLOOR_BYTES). Ties keep the smaller k (fewer combine members).
    """
    tw, trt = geo.tensor_w_tiles, geo.tensor_row_tiles
    best_k, best = k_fit, _occ_cost(trt, tw, _normalize_k(tw, k_fit)[0], grid_x, grid_y)
    if tw < 2 * OCC_TILES_PER_MEMBER:
        return best_k
    for cand in range(k_fit + 1, k_cap + 1):
        kk, s = _normalize_k(tw, cand)
        if s < OCC_TILES_PER_MEMBER:
            break
        if sz.is_rm and s * TILE * elem_bytes < RM_PIECE_FLOOR_BYTES:
            break
        if _max_groups(kk, grid_x, grid_y) < 1:
            break
        if footprint(sz, tw, kk, 1, 1, PlanOpts(owner_combine=kk > K_SYM_MAX_INTERLEAVED)) > budget:
            continue  # the owner's k gathered partials outgrow L1
        c = _occ_cost(trt, tw, kk, grid_x, grid_y)
        if c < best:
            best_k, best = kk, c
    return best_k


def select_blocking(geo: Geometry, sz: _Sizes, grid_x: int, grid_y: int, budget: int, elem_bytes: int) -> Blocking:
    """INTERLEAVED: row-groups over the whole grid x a W split over k consecutive Tensix cores (one grid row while
    k <= grid_x, else row-major across grid rows)."""
    k_cap = min(K_CAP_MAX, grid_x * grid_y)
    tw, trt = geo.tensor_w_tiles, geo.tensor_row_tiles

    k_fit = None
    for k in range(1, k_cap + 1):
        if footprint(sz, tw, k, 1, 1) <= budget:
            k_fit = k
            break
    if k_fit is None:
        raise RuntimeError(
            f"rms_norm: W={geo.W} too wide — even w_split_cores={k_cap} does not fit L1 ({budget} B per core)"
        )

    k, slice_w = _normalize_k(tw, _choose_k(geo, sz, grid_x, grid_y, k_fit, k_cap, elem_bytes, budget))
    c, pitch = _pitch(sz, slice_w)

    num_groups = min(_max_groups(k, grid_x, grid_y), trt)
    base, rem = divmod(trt, num_groups)

    def core_of(g, m):
        if k <= grid_x:
            gpr = grid_x // k
            return ttnn.CoreCoord((g % gpr) * k + m, g // gpr)
        idx = g * k + m
        return ttnn.CoreCoord(idx % grid_x, idx // grid_x)

    groups, start = [], 0
    for g in range(num_groups):
        n = base + (1 if g < rem else 0)
        members = []
        for m in range(k):
            w_start = m * slice_w
            core_w = min(slice_w, tw - w_start)
            col0 = w_start * TILE
            members.append(Member(core_of(g, m), m, start, n, w_start, core_w, col0, min(core_w * TILE, geo.W - col0)))
        groups.append(members)
        start += n
    crt_max = base + (1 if rem else 0)

    b, d, opts = _solve_rows(
        sz,
        k,
        slice_w,
        crt_max,
        budget,
        PlanOpts(owner_combine=k > K_SYM_MAX_INTERLEAVED),
        allow_depth2=not sz.is_rm,
        overlap_blocks=1 if sz.is_rm else OVERLAP_BLOCKS,
    )
    return Blocking(
        k=k,
        slice_w_tiles=slice_w,
        rm_chunk_w_tiles=c,
        x_row_pitch=pitch,
        groups=groups,
        block_row_tiles=b,
        x_block_depth=d,
        out_unit_tiles=_out_unit(slice_w),
        opts=opts,
        footprint=_plan_bytes(cb_plan(sz, k, slice_w, b, d, opts)),
    )


def _ordered_shard_cores(grid, row_major: bool):
    """Cores of a shard grid in shard-assignment order (row-major or column-major)."""
    cores = set()
    for r in grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.add((x, y))
    return sorted(cores, key=(lambda c: (c[1], c[0])) if row_major else (lambda c: (c[0], c[1])))


def select_sharded_blocking(geo: Geometry, sz: _Sizes, input_tensor, budget: int) -> Blocking:
    """HEIGHT / WIDTH / BLOCK_SHARDED: the shard spec IS the work split; block = the resident shard.

    Groups = shard rows (WIDTH: one), k = shard columns (HEIGHT: one — local reduction, no combine). Member -> (slice, core) from the shard grid in its
    orientation: WIDTH walks the grid's cores in order (a group may span grid rows); BLOCK puts shard
    (row g, column i) at (x0 + i, y0 + g) for ROW_MAJOR, (x0 + g, y0 + i) for COL_MAJOR. Ragged last shard
    row / column (Split.PAD): the members own only their valid tile-rows / W tiles.
    """
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    if sz.is_rm:
        rows_ext, unit_h = geo.num_images * geo.H, sh  # sticks
        w_ext, unit_w = geo.W, sw  # datums
        slice_w = _div_up(sw, TILE)
    else:
        if sh % TILE or sw % TILE:
            raise RuntimeError(f"rms_norm: TILE shard {sh}x{sw} is not a whole number of tiles")
        rows_ext, unit_h = geo.tensor_row_tiles, sh // TILE  # tile-rows
        w_ext, unit_w = geo.tensor_w_tiles, sw // TILE  # tiles
        slice_w = unit_w
    n_h, n_w = _div_up(rows_ext, unit_h), _div_up(w_ext, unit_w)

    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if n_h != 1:
            raise RuntimeError(f"rms_norm: WIDTH shard height {sh} does not cover the tensor")
        order = _ordered_shard_cores(ss.grid, row_major)
        if len(order) < n_w:
            raise RuntimeError(f"rms_norm: {n_w} W shards but only {len(order)} shard-grid cores")

        def core_of(g, i):
            return order[i]

    elif mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # k = 1: every Tensix core holds whole rows (the reduction is local, no combine). Shard row g lives on the
        # g-th core of the grid in its orientation (a multi-range grid, e.g. full rows + a partial last row).
        if n_w != 1:
            raise RuntimeError(f"rms_norm: HEIGHT shard width {sw} does not cover the tensor")
        order = _ordered_shard_cores(ss.grid, row_major)
        if len(order) < n_h:
            raise RuntimeError(f"rms_norm: {n_h} row shards but only {len(order)} shard-grid cores")

        def core_of(g, i):
            return order[g]

    else:
        bb = ss.grid.bounding_box()
        x0, y0 = bb.start.x, bb.start.y

        def core_of(g, i):
            return (x0 + i, y0 + g) if row_major else (x0 + g, y0 + i)

    k = n_w
    groups = []
    for g in range(n_h):
        members = []
        for i in range(k):
            x, y = core_of(g, i)
            if sz.is_rm:
                stick0 = g * sh
                sticks = min(sh, rows_ext - stick0)
                col0 = i * sw
                valid_cols = min(sw, geo.W - col0)
                core_w = _div_up(valid_cols, TILE)
                mem = Member(
                    ttnn.CoreCoord(x, y), i, 0, _div_up(sticks, TILE), col0 // TILE, core_w, col0, valid_cols, sticks
                )
            else:
                row0 = g * unit_h
                w0 = i * unit_w
                core_w = min(unit_w, geo.tensor_w_tiles - w0)
                col0 = w0 * TILE
                mem = Member(
                    ttnn.CoreCoord(x, y),
                    i,
                    row0,
                    min(unit_h, rows_ext - row0),
                    w0,
                    core_w,
                    col0,
                    min(core_w * TILE, geo.W - col0),
                )
            members.append(mem)
        groups.append(members)
    crt_max = max(grp[0].core_rows for grp in groups)

    base_opts = PlanOpts(
        shard_backed=not sz.is_rm,
        owner_combine=k > K_SYM_MAX_SHARDED,
        mask_tail=(not sz.is_rm) and geo.W % TILE != 0,
    )
    b, d, opts = _solve_rows_sharded(sz, k, slice_w, crt_max, budget, base_opts)
    c, pitch = _pitch(sz, slice_w, opts.rm_chunk_cap)
    return Blocking(
        k=k,
        slice_w_tiles=slice_w,
        rm_chunk_w_tiles=c,
        x_row_pitch=pitch,
        groups=groups,
        block_row_tiles=b,
        x_block_depth=d,
        out_unit_tiles=_out_unit(slice_w),
        opts=opts,
        footprint=_plan_bytes(cb_plan(sz, k, slice_w, b, d, opts)),
    )


def _gamma_mcast_groups(blk: Blocking, grid_x: int):
    """(receiver_group_size, core order) of the interleaved gamma multicast, or None (per-core gamma load).

    ttnn.Mcast cuts the ordered active cores into equal consecutive groups, and a group must share one W slice.
    k == 1: every Tensix core holds the same slice -> one group of all of them. k > 1: member m of row-group g sits
    in grid column (g % gpr) * k + m, so a grid column holds a single slice; when the row-groups fill whole grid
    rows every column has the same count -> one group per column (column-major order). Otherwise (a partial last
    grid row of row-groups) the columns differ in size and each Tensix core loads its own slice.
    """
    n = len(blk.groups)
    if blk.k == 1:
        return n, ttnn.McastCoreOrder.RowMajor
    if blk.k > grid_x:
        return None  # groups span grid rows: a slice's Tensix cores are not one grid column
    gpr = grid_x // blk.k
    if n % gpr == 0:
        return n // gpr, ttnn.McastCoreOrder.ColumnMajor
    return None


def is_sharded(tensor) -> bool:
    """Any shard scheme (HEIGHT / WIDTH / BLOCK): the shard spec fixes the work split, consumed in place."""
    return tensor.memory_config().is_sharded()


def cb_l1_budget(device) -> int:
    """Bytes of L1 per Tensix core the program's CBs may occupy.

    CBs are placed bottom-up from the allocator's L1 base; L1 buffers (interleaved-L1 / sharded
    tensors, incl. this op's own output) are allocated top-down. The budget is therefore the free
    region between the allocator base and the lowest allocated L1 buffer. NOT
    `ttnn.get_max_worker_l1_unreserved_size()`: that figure starts at the kernel-config ring buffer,
    below the allocator base, and overshoots by its size (~69 KB on BH) — shapes whose footprint fell
    in that gap threw "circular buffers grow to N B which is beyond max L1 size".
    """
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else int(view.total_bytes_per_bank)


def _datum_bytes(tensor) -> int:
    """Bytes per datum in L1 (bfloat8_b: one mantissa byte — Tensor.element_size() throws for block-float)."""
    return 1 if tensor.dtype == ttnn.bfloat8_b else tensor.element_size()


def _bfp_exp_section_bytes(dtype, elem_bytes: int) -> int:
    """Bytes of shared-exponent section ahead of the datums in one tile of `dtype` (0 for plain formats).

    A bfloat8_b tile is [one exponent byte per 16-datum face row (64 B)][one mantissa byte per datum]; the
    datum area is TILE*TILE*elem_bytes, so the section is whatever the tile size holds beyond it.
    """
    return ttnn.tile_size(dtype) - TILE * TILE * elem_bytes


def _cb(index, pages, page_size, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _shard_backed_cb(index, tensor, core_ranges):
    """CB aliasing a TILE tensor's resident shard (zero-copy: the shard IS the CB's L1)."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    cb.core_ranges = core_ranges
    return cb


def create_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    device = input_tensor.device()
    geo = derive_geometry(input_tensor.shape)
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    sharded = is_sharded(input_tensor)
    has_gamma = gamma is not None
    x_dtype = input_tensor.dtype
    g_dtype = gamma.dtype if has_gamma else x_dtype
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    t_acc = acc_dtype(fp32_dest)
    sz = _Sizes(x_dtype, g_dtype, is_rm, has_gamma, fp32_dest)
    elem_bytes = _datum_bytes(input_tensor)
    g_elem_bytes = _datum_bytes(gamma) if has_gamma else elem_bytes
    gamma_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT

    budget = cb_l1_budget(device)
    grid = device.compute_with_storage_grid_size()
    if sharded:
        blk = select_sharded_blocking(geo, sz, input_tensor, budget)
    else:
        blk = select_blocking(geo, sz, grid.x, grid.y, budget, elem_bytes)
    k, C, pitch = blk.k, blk.rm_chunk_w_tiles, blk.x_row_pitch
    b, d = blk.block_row_tiles, blk.x_block_depth
    opts = blk.opts
    combine_lookahead = 1 if (k > 1 and d == 2 and not opts.owner_combine) else 0

    # ---- active Tensix cores: every member of every group (identical CB list on all of them: peers address
    #      cb_gathered_sumsq / cb_inv_rms by their own write pointer) ----
    all_cores = [mem.core for grp in blk.groups for mem in grp]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores])

    # ---- circular buffers ----
    cb_dtype = {
        CB_X: x_dtype,
        CB_X_ALT: x_dtype,
        CB_GAMMA: g_dtype,
        CB_SUMSQ: t_acc,
        CB_PARTIAL: t_acc,
        CB_GATHERED: t_acc,
        CB_INV_RMS: t_acc,
        CB_X_RM: x_dtype,
        CB_OUT: x_dtype,
        CB_OUT_RM: x_dtype,
        CB_SUMSQ_ACC: t_acc,
        CB_TAIL_MASK: t_acc,
        CB_INV_OWNED: t_acc,
    }
    plan = cb_plan(sz, k, blk.slice_w_tiles, b, d, opts)
    cbs = [_cb(idx, pages, page_bytes, cb_dtype[idx], core_ranges) for idx, (pages, page_bytes) in plan.items()]
    if opts.shard_backed:
        cbs.append(_shard_backed_cb(CB_X, input_tensor, core_ranges))
        cbs.append(_shard_backed_cb(CB_OUT, output_tensor, core_ranges))
        out_cb_pages = 0  # TILE sharded writer drains per block; no ring clamp
    else:
        out_cb_pages = plan[CB_OUT][0]

    defines = []
    if is_rm:
        defines.append(("RMS_RM", "1"))
    if has_gamma:
        defines.append(("RMS_HAS_GAMMA", "1"))
    if sharded:
        defines.append(("RMS_SHARDED", "1"))
    if opts.mask_tail:
        defines.append(("RMS_MASK_TAIL", "1"))
    gamma_groups = None
    if opts.gamma_chunk:
        defines.append(("RMS_GAMMA_CHUNK", str(opts.gamma_chunk)))  # streamed gamma: per-core chunk ring, no mcast
        defines.append(("RMS_GAMMA_STREAM_DEPTH", str(GAMMA_STREAM_DEPTH)))
    elif has_gamma and (GAMMA_MCAST_SHARDED_K1 and k == 1 if sharded else GAMMA_MCAST):
        gamma_groups = _gamma_mcast_groups(blk, grid.x)
    gamma_mcast = gamma_groups is not None
    if gamma_mcast:
        defines.append(("RMS_GAMMA_MCAST", "1"))

    x_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0

    # Sharded: the local shard is consumed in place, never through an accessor -> empty accessor args.
    x_acc_args = ttnn.TensorAccessorArgs() if sharded else ttnn.TensorAccessorArgs(input_tensor)
    out_acc_args = ttnn.TensorAccessorArgs() if sharded else ttnn.TensorAccessorArgs(output_tensor)

    reader_ct = [geo.tensor_w_tiles, geo.W, geo.H, geo.image_row_tiles, pitch, C, x_page, elem_bytes, d]
    reader_ct.extend(x_acc_args.get_compile_time_args())

    writer_ct = [
        geo.tensor_w_tiles,
        geo.W,
        geo.H,
        geo.image_row_tiles,
        C,
        out_page,
        elem_bytes,
        k,
        blk.out_unit_tiles,
        SEM_PARTIALS_ARRIVED,
        out_cb_pages,
        WRITER_NOC_XY_ARG_BASE,
        1 if opts.owner_combine else 0,
        1 if PARTIAL_FACES_ONLY else 0,
        SEM_INV_ARRIVED,
        pitch,
        g_page,
        g_elem_bytes,
        1 if gamma_is_tile else 0,
        _bfp_exp_section_bytes(g_dtype, g_elem_bytes) if has_gamma else 0,
        1 if (GAMMA_AFTER_SEND and b <= GAMMA_AFTER_SEND_MAX_ROWS) else 0,
        1 if SIGNAL_AFTER_FLUSH else 0,
    ]
    writer_ct.extend(out_acc_args.get_compile_time_args())
    writer_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    compute_ct = [k, pitch, C, d, combine_lookahead, 1 if opts.owner_combine else 0]

    inv_w_bits = _f32_bits(1.0 / geo.W)
    eps_bits = _f32_bits(epsilon)
    x_addr = input_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    out_addr = output_tensor.buffer_address()

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tw = geo.tensor_w_tiles
    for members in blk.groups:
        noc_xy = []
        for mem in members:
            v = device.worker_core_from_logical_core(mem.core)
            noc_xy.extend([v.x, v.y])
        for mem in members:
            c = mem.core
            core_rows = mem.core_rows
            num_blocks = _div_up(core_rows, b)
            last_rows = core_rows - (num_blocks - 1) * b
            chunks_this = _div_up(mem.core_w, C)
            owns_tail = 1 if (opts.mask_tail and mem.w_tile_start + mem.core_w == tw) else 0
            # TILE sharded: every tile-row pushes the whole shard row (pitch) into the shard-backed cb_out.
            norm_w = pitch if opts.shard_backed else mem.core_w
            reader_rt[c.x][c.y] = [
                x_addr,
                mem.row_start,
                num_blocks,
                b,
                last_rows,
                mem.w_tile_start,
                mem.core_w,
                chunks_this,
                mem.valid_cols,
                mem.valid_sticks,
                owns_tail,
            ]
            compute_rt[c.x][c.y] = [
                num_blocks,
                b,
                last_rows,
                mem.core_w,
                chunks_this,
                inv_w_bits,
                eps_bits,
                mem.m,
                norm_w,
                owns_tail,
            ]
            writer_base = [
                out_addr,
                mem.row_start,
                num_blocks,
                b,
                last_rows,
                mem.w_tile_start,
                mem.core_w,
                chunks_this,
                mem.m,
                mem.valid_cols,
                mem.valid_sticks,
                g_addr,
                mem.col0,
            ]
            assert len(writer_base) == WRITER_NOC_XY_ARG_BASE
            writer_rt[c.x][c.y] = writer_base + noc_xy

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        defines=defines,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        defines=defines,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        defines=defines,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    semaphores = []
    if k > 1:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_PARTIALS_ARRIVED, core_ranges=core_ranges, initial_value=0))
        if opts.owner_combine:
            semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_INV_ARRIVED, core_ranges=core_ranges, initial_value=0))
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    if gamma_mcast:
        # Sender = each group's first Tensix core (reads + expands the slice); one-shot, no handshake.
        group_size, order = gamma_groups
        mc = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=False),  # the writer's NoC (WriterConfigDescriptor)
            core_ranges,
            group_size,
            ttnn.McastFixedSenderConfig(),
            order,
        )
        mc.attach(descriptor, "gamma_mcast", [writer], SEM_GAMMA_MCAST_FIRST)
    descriptor.kernels = [reader, writer, compute]
    return descriptor
