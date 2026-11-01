# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: geometry, select_blocking(), CBs, kernels, per-core args.

Blocking (op_design.md -> Blocking Model):
  * row axis (tile-rows, leading dims folded): split over row-groups across the whole Tensix grid;
    each group's assignment is processed in blocks of `block_row_tiles` tile-rows.
  * W axis: split over `w_split_cores` (k) consecutive Tensix cores of one grid row; every member
    keeps its W share resident (cb_x) across pass 1 and pass 2, and the k members all-gather their
    per-row sum-of-squares partials over the NoC (k > 1 only).
Every knob below (k, block_row_tiles, x_block_depth, rm_chunk_w_tiles, out_unit_tiles) is computed
once here and handed to the kernels as a CT/RT arg; the kernels derive all trip counts from them.

WIDTH / BLOCK_SHARDED (Refinement 2): the shard spec fixes the work split — one group per shard row
(BLOCK) or a single group (WIDTH), k = shard columns, members and their NoC coordinates from the shard
grid (orientation-aware, groups may span grid rows). Block = the resident shard (sub-blocked along rows
only under L1 pressure). TILE: cb_x / cb_out are backed on the input / output shards (zero-copy). RM:
the resident sticks are copied locally into the tilize staging CB (never gathered across Tensix cores).
Combine topology by k: k <= K_SYM_MAX symmetric unicast all-gather; beyond, a per-row owner gathers the
k partials, combines + finalizes, and returns inv_rms to every member (L1 per member ~ k * ceil(b/k)).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
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
CB_INV_OWNED = 12  # owner combine (k > K_SYM_MAX): inv_rms of the tile-rows this member owns

SEM_PARTIALS_ARRIVED = 0
SEM_INV_ARRIVED = 1  # owner combine: inv_rms tiles returned by the row owners
WRITER_NOC_XY_ARG_BASE = 11  # writer RT index of the first peer (x, y) pair — passed to the kernel as a CT arg

# Knob constants (design Phase 0 values).
K_CAP_MAX = 8  # unicast all-gather mechanism cap (interleaved W split)
K_SYM_MAX = 8  # combine topology: symmetric all-gather up to k members, per-row owner gather + return beyond
PARTIAL_FACES_ONLY = True  # ship faces 0/2 of a partial / inv_rms tile (column 0 lives there), not the whole tile
OUT_DEPTH = 2  # cb_out / cb_out_rm buffer depth
RM_IN_DEPTH = 2  # cb_x_rm buffer depth
OUT_UNIT_CAP = 16  # TILE writer batch (tiles per wait/flush/pop)
RM_CHUNK_CAP = 32  # balanced RM streaming chunk cap (tiles)
RM_PIECE_FLOOR_BYTES = 512  # occupancy split must keep RM stick pieces >= this
OCC_TILES_PER_MEMBER = 4  # occupancy split keeps >= 4 W tiles per member
OCC_ROWS_FRACTION = 8  # occupancy split only when 8 * tensor_row_tiles <= G


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


def _rm_chunk(slice_w_tiles: int):
    chunks = _div_up(slice_w_tiles, RM_CHUNK_CAP)
    c = _div_up(slice_w_tiles, chunks)
    return c, chunks * c


@dataclass(frozen=True)
class PlanOpts:
    """Regime switches that change the CB inventory (one source of truth for footprint() and the CB list)."""

    shard_backed: bool = False  # TILE sharded: cb_x / cb_out alias the shards (tensor buffers, not CB budget)
    owner_combine: bool = False  # k > K_SYM_MAX: per-row owner gather + return instead of the all-gather
    gather_depth: int = 2  # symmetric all-gather: blocks of cb_gathered (1 when every core runs one block)
    mask_tail: bool = False  # sharded TILE, W % 32 != 0: tail tile's x^2 masked inside the collapse reduce


def cb_plan(sz: _Sizes, k: int, slice_w: int, b: int, d: int, opts: PlanOpts = PlanOpts()) -> dict:
    """Single source of truth for every CB's capacity: {cb_index: (pages, page_bytes)}.

    Both footprint() (the L1-fit solve in select_blocking) and create_program_descriptor() (the CB
    declarations) read this dict, so a knob change lands in exactly one place (l1_ledger.md rows).
    """
    if sz.is_rm:
        c, pitch = _rm_chunk(slice_w)
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
        plan[CB_GAMMA] = (pitch, sz.T_g)
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


def _pitch(sz: _Sizes, slice_w: int):
    """(rm_chunk_w_tiles, x_row_pitch) for a slice of `slice_w` tiles."""
    return _rm_chunk(slice_w) if sz.is_rm else (1, slice_w)


def _solve_rows(sz: _Sizes, k: int, slice_w: int, crt_max: int, budget: int, opts: PlanOpts, allow_depth2: bool):
    """Largest block_row_tiles (and x_block_depth) whose CB plan fits `budget`.

    One block per core (the whole assignment) when it fits; else the largest sub-block at depth 2 (TILE
    interleaved) / 1. The symmetric all-gather needs a 2-block cb_gathered only when a core runs > 1 block.
    """

    def fits(b, d):
        single = b >= crt_max
        o = PlanOpts(opts.shard_backed, opts.owner_combine, 1 if single else 2, opts.mask_tail)
        return _plan_bytes(cb_plan(sz, k, slice_w, b, d, o)) <= budget, o

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


def select_blocking(geo: Geometry, sz: _Sizes, grid_x: int, grid_y: int, budget: int, elem_bytes: int) -> Blocking:
    """INTERLEAVED: row-groups over the whole grid x a W split over k consecutive cores of one grid row."""
    G = grid_x * grid_y
    k_cap = min(K_CAP_MAX, grid_x)
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

    k_occ = 1
    if OCC_ROWS_FRACTION * trt <= G and tw >= 8:
        k_occ = min(k_cap, G // trt, tw // OCC_TILES_PER_MEMBER)
        if sz.is_rm:
            k_occ = min(k_occ, max(1, (geo.W * elem_bytes) // RM_PIECE_FLOOR_BYTES))
        k_occ = max(1, k_occ)
    k, slice_w = _normalize_k(tw, max(k_fit, k_occ))
    c, pitch = _pitch(sz, slice_w)

    groups_per_grid_row = grid_x // k
    num_groups = min(groups_per_grid_row * grid_y, trt)
    base, rem = divmod(trt, num_groups)
    groups, start = [], 0
    for g in range(num_groups):
        n = base + (1 if g < rem else 0)
        vx, vy = g % groups_per_grid_row, g // groups_per_grid_row
        members = []
        for m in range(k):
            w_start = m * slice_w
            core_w = min(slice_w, tw - w_start)
            col0 = w_start * TILE
            members.append(
                Member(
                    ttnn.CoreCoord(vx * k + m, vy), m, start, n, w_start, core_w, col0, min(core_w * TILE, geo.W - col0)
                )
            )
        groups.append(members)
        start += n
    crt_max = base + (1 if rem else 0)

    b, d, opts = _solve_rows(sz, k, slice_w, crt_max, budget, PlanOpts(), allow_depth2=not sz.is_rm)
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
    """WIDTH / BLOCK_SHARDED: the shard spec IS the work split; block = the resident shard.

    Groups = shard rows (WIDTH: one), k = shard columns. Member -> (slice, core) from the shard grid in its
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

    c, pitch = _pitch(sz, slice_w)
    base_opts = PlanOpts(
        shard_backed=not sz.is_rm,
        owner_combine=k > K_SYM_MAX,
        mask_tail=(not sz.is_rm) and geo.W % TILE != 0,
    )
    b, d, opts = _solve_rows(sz, k, slice_w, crt_max, budget, base_opts, allow_depth2=False)
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


def is_sharded_wb(tensor) -> bool:
    return tensor.memory_config().memory_layout in (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )


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
    sharded = is_sharded_wb(input_tensor)
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
    if sharded:
        blk = select_sharded_blocking(geo, sz, input_tensor, budget)
    else:
        grid = device.compute_with_storage_grid_size()
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

    x_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0

    # Sharded: the local shard is consumed in place, never through an accessor -> empty accessor args.
    x_acc_args = ttnn.TensorAccessorArgs() if sharded else ttnn.TensorAccessorArgs(input_tensor)
    out_acc_args = ttnn.TensorAccessorArgs() if sharded else ttnn.TensorAccessorArgs(output_tensor)

    reader_ct = [
        geo.tensor_w_tiles,
        geo.W,
        geo.H,
        geo.image_row_tiles,
        pitch,
        C,
        x_page,
        g_page,
        elem_bytes,
        g_elem_bytes,
        1 if gamma_is_tile else 0,
        d,
        _bfp_exp_section_bytes(g_dtype, g_elem_bytes) if has_gamma else 0,
    ]
    reader_ct.extend(x_acc_args.get_compile_time_args())
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

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
    ]
    writer_ct.extend(out_acc_args.get_compile_time_args())

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
                g_addr,
                mem.row_start,
                num_blocks,
                b,
                last_rows,
                mem.w_tile_start,
                mem.core_w,
                chunks_this,
                mem.col0,
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
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
