# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor.

`select_blocking()` is the single source of every blocking knob (op_design.md -> Work Distribution); every CB page
count, loop trip count, RT arg and Mcast group size below is derived from its result.

Regimes (both built):
  LOCAL   (w_split == 1): R (tile-rows) split over lines of one core; whole row resident per core.
  W_SPLIT (w_split  > 1): lines of `w_split` consecutive cores (row-major) share an R-range; each core holds its
                          W-slice resident; Σx² partials are gathered to the line root (cores[0] of the line), the
                          root combines + finalizes rinv and multicasts it back to every core of the line.

Placements (Refinement 2): an INTERLEAVED input lets select_blocking() choose the cores; a sharded input fixes them —
the shard grid IS the core assignment (`ShardPlacement`): HEIGHT -> one LOCAL line per shard core, WIDTH -> one W_SPLIT
line over the whole shard grid, BLOCK -> one W_SPLIT line per shard-grid row (column for COL_MAJOR orientation).
TILE shards are consumed in place: cb_x / cb_y are backed zero-copy on the input / output shard. ROW_MAJOR shards are
re-strided core-locally (L1 -> L1) into cb_x_rm and back out of cb_out_rm (tilize needs a `core_w_tiles·32` stride).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FACE = 16

# ---- knob constants (one definition each; op_design.md Work Distribution) ----
L1_CB_BUDGET_BYTES = 1 << 20
MIN_CORE_W_TILES = 8
X_BLOCK_DEPTH = 2  # cb_x (TILE) depth in blocks
X_RM_DEPTH = 2  # cb_x_rm depth in tile-rows
Y_CHUNK_TILES = 8  # cb_y writer chunk
Y_DEPTH = 2  # cb_y depth in chunks
OUT_RM_DEPTH = 2  # cb_out_rm depth in tile-rows
READ_CHUNK_TILES = 8  # reader tiles per barrier+push (TILE)
GAMMA_CHUNK_DEPTH = 2  # cb_gamma / cb_gamma_rm depth in W-chunks when γ streams (w_chunk_tiles < core_w_tiles)
SHARDED_CB_MARGIN_BYTES = 16 << 10  # slack below the resident shard buffers (CB alignment / rounding)
# W_SPLIT rendezvous knobs (Refinement 3; op_design.md lamps L4 / L5):
GATHER_COL_FACES = True  # L4: gather only faces 0 + 2 of each column-0-valid partial (False: whole tiles)
# L5: no receiver-ready round trip on the rinv multicast. Safe by the causal chain: a core sends partial(b+1) only
# after its normalize(b) consumed cb_rinv(b) (and its reader cleared round b's data-ready flag), and the root sends
# rinv(b+1) only after every partial(b+1) landed — so the landing slot and the flag are always free.
RINV_MCAST_HANDSHAKE = False
# x·γ prescale into cb_xg before the rinv wait (W_SPLIT TILE γ; per-shape gate in _xg_prescale). False: the fused
# normalize (x·γ)·rinv after the rendezvous (pre-Refinement-3 schedule).
XG_PRESCALE = True
# Per-core compute-throughput knobs (Refinement 4; catalog `compute_block_size`):
# DEST tiles per eltwise_chain window for the normalize / x·γ prescale passes (block lifecycle on the output CB:
# one acquire / commit / pack / release per NORM_DEST_BLOCK tiles instead of per tile; the chain clamps it to
# DEST_AUTO_LIMIT). 1 = the per-tile schedule. Applied only where the output CB's ring can never be straddled by
# a window (_norm_block).
NORM_DEST_BLOCK = 8
# Collapse Σx² tiles to the per-row column on the ReduceTile datapath (FPU reduce against a ones scaler in
# CB_REDUCE_SCALER) instead of AccumulateViaAdd's SFPU lane reduce (~6x fewer cycles per tile). The masked
# partial-W collapse (resident non-aligned W) keeps AccumulateViaAdd (its 0/1 lane mask is that datapath's).
COLLAPSE_REDUCE_TILE = True

# ---- CB indices (semantic names) ----
CB_X_RM = 0
CB_X = 1
CB_GAMMA_RM = 2
CB_GAMMA = 3
CB_SUMSQ_RAW = 4
CB_PARTIAL = 5
CB_GATHERED = 6
CB_RINV_COL = 7
CB_RINV_BCAST = 8
CB_RINV = 9
CB_Y = 10
CB_OUT_RM = 11
CB_WMASK = 12  # sharded TILE + non-aligned W: 0/1 lane mask for the partial-W tile (collapse reduce scaler slot)
CB_GAMMA_STAGE = 13  # RM γ whose slice offset is not DRAM-aligned (RM shards): aligned DRAM window staging
CB_XG = 14  # W_SPLIT TILE γ: x·γ of the block, computed while waiting for rinv (Blocking.xg_prescale)
CB_REDUCE_SCALER = 15  # ones scaler of the ReduceTile collapse (COLLAPSE_REDUCE_TILE)

# ---- semaphore ids ----
SEM_PARTIALS = 0  # root: count of partials landed in cb_gathered (cumulative)
SEM_RINV_LOCAL = 1  # root: root writer -> root reader "rinv(b) landed in my own cb_rinv" (cumulative)
SEM_MCAST_FIRST = 2  # Mcast channel semaphores start here
MCAST_PREFIX = "rinv_mcast"


def _div_up(a, b):
    return (a + b - 1) // b


def _element_bytes(t) -> int:
    """Element bytes for addressing. Block-float (bfloat8_b) has no per-element byte size (Tensor.element_size()
    raises); report its 1-byte mantissa — every element-addressed path (RM sticks, pad-lane zeroing, face-row γ
    reads) is host-guarded off for block-float, and the kernels static_assert it."""
    return 1 if t.dtype == ttnn.bfloat8_b else t.element_size()


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@dataclass(frozen=True)
class OpGeometry:
    is_tile: bool
    has_gamma: bool
    gamma_is_rm: bool
    t_in: int  # input (= output) tile bytes
    t_g: int  # gamma tile bytes
    es_g: int  # gamma element bytes
    t_int: int  # intermediate-CB tile bytes (Float32 when fp32 DEST is on, Float16_b when off)
    x_resident: bool = False  # TILE sharded input: cb_x backed on the shard (no allocator budget)
    y_resident: bool = False  # TILE sharded output: cb_y backed on the shard (no allocator budget)
    sumsq_parts: int = 1  # raw Σx² tiles per tile-row (2 when a resident partial-W tile is squared apart + masked)
    gamma_staged: bool = False  # cb_gamma_stage present (RM γ slice offset off the DRAM read grain)
    dram_align: int = 64


@dataclass(frozen=True)
class Blocking:
    regime: str  # "LOCAL" | "W_SPLIT"
    w_split: int
    core_w_tiles: int
    num_lines: int
    block_rows: int
    # W-chunk extent of the per-row staging (RM cb_x_rm / cb_out_rm) and of γ (cb_gamma / cb_gamma_rm). == core_w_tiles
    # (no chunking) unless a sharded row is too wide to stage whole next to its resident shards; then γ streams in
    # chunks per block and RM tilize / untilize run chunk by chunk (core_w_tiles is padded to a chunk multiple).
    w_chunk_tiles: int = 0
    # W_SPLIT stall shadow (Refinement 3): compute x·γ into cb_xg (one block, intermediate format) before the rinv
    # wait, so only y = xg·rinv is left after the rendezvous. Enabled only where cb_xg fits next to the chosen
    # block (it never shrinks block_rows).
    xg_prescale: bool = False

    @property
    def gamma_streams(self) -> bool:
        return self.w_chunk_tiles < self.core_w_tiles


@dataclass(frozen=True)
class ShardPlacement:
    """The core assignment a sharded input fixes (op_design.md -> Sharded placements). Rows are tile-rows (TILE) or
    sticks (ROW_MAJOR); `lines[i]` lists the cores of line i in W order (position 0 = the line root)."""

    lines: tuple  # tuple[tuple[CoreCoord, ...], ...]
    line_units: tuple  # per line: (first row unit, valid row units)  -- tile-rows (TILE) / sticks (RM)
    shard_w: int  # shard width in elements
    core_w_tiles: int  # ⌈shard_w / 32⌉
    w_split: int  # cores per line
    col_major: bool  # COL_MAJOR orientation (lines are shard-grid columns)


def _collapse_reduce_tile(g: OpGeometry) -> bool:
    return COLLAPSE_REDUCE_TILE and g.sumsq_parts == 1


def _norm_block(is_tile: bool, y_resident: bool, br: int, cw: int) -> int:
    """DEST window of the normalize / prescale chains. A window reserves its tiles contiguously, so it must never
    straddle the end of the output CB ring: the resident output shard and cb_xg are consumed whole per block; the
    interleaved cb_y ring is safe when both the ring and the block are window multiples. RM normalizes in place
    with per-tile credit (window 1)."""
    if not is_tile or NORM_DEST_BLOCK <= 1:
        return 1
    if y_resident:
        return NORM_DEST_BLOCK
    ring = Y_DEPTH * Y_CHUNK_TILES
    return NORM_DEST_BLOCK if (ring % NORM_DEST_BLOCK == 0 and (br * cw) % NORM_DEST_BLOCK == 0) else 1


def _per_row_bytes(g: OpGeometry, cw: int, k: int) -> int:
    ws = k > 1
    b = 0
    if not g.x_resident:
        b += (X_BLOCK_DEPTH * cw * g.t_in) if g.is_tile else (cw * g.t_in)  # cb_x
    b += (g.sumsq_parts + 2) * g.t_int  # cb_sumsq_raw (parts per tile-row), cb_rinv_col, cb_rinv
    if ws:
        b += (2 + k) * g.t_int  # cb_partial, cb_rinv_bcast, cb_gathered
    return b


def _gamma_depth(cw: int, wc: int) -> int:
    return 1 if wc >= cw else GAMMA_CHUNK_DEPTH


def _gamma_stage_bytes(g: OpGeometry, wc: int) -> int:
    return wc * TILE * g.es_g + 2 * g.dram_align if g.gamma_staged else 0


def _fixed_bytes(g: OpGeometry, cw: int, wc: int = None) -> int:
    wc = cw if wc is None else wc
    b = 0
    if g.has_gamma:
        gd = _gamma_depth(cw, wc)
        b += gd * wc * g.t_g  # cb_gamma
        if g.gamma_is_rm:
            b += gd * wc * TILE * g.es_g  # cb_gamma_rm
    if g.is_tile:
        if not g.y_resident:
            b += Y_DEPTH * Y_CHUNK_TILES * g.t_in  # cb_y
    else:
        b += X_RM_DEPTH * wc * g.t_in + OUT_RM_DEPTH * wc * g.t_in  # cb_x_rm, cb_out_rm
    if g.sumsq_parts > 1:
        b += g.t_int  # cb_wmask
    if _collapse_reduce_tile(g):
        b += ttnn.tile_size(ttnn.bfloat16)  # cb_reduce_scaler
    b += _gamma_stage_bytes(g, wc)  # cb_gamma_stage
    return b


def _footprint(g: OpGeometry, br: int, cw: int, k: int) -> int:
    return br * _per_row_bytes(g, cw, k) + _fixed_bytes(g, cw)


def _round_up(a, b):
    return _div_up(a, b) * b


def _xg_prescale(g: OpGeometry, br: int, cw: int, wc: int, k: int, budget: int) -> bool:
    """x·γ prescale (cb_xg, br·cw intermediate tiles) for a W_SPLIT TILE line with resident γ, if it fits."""
    if not (XG_PRESCALE and k > 1 and g.is_tile and g.has_gamma and wc >= cw):
        return False
    return br * _per_row_bytes(g, cw, k) + _fixed_bytes(g, cw, wc) + br * cw * g.t_int <= budget


def select_blocking(
    g: OpGeometry, R: int, Wt: int, C: int, shard: ShardPlacement = None, budget: int = L1_CB_BUDGET_BYTES
) -> Blocking:
    """The single source of every knob (op_design.md select_blocking).

    Interleaved: chooses w_split / core_w_tiles / num_lines over the grid. Sharded: the shard spec fixes them
    (w_split = cores per line, core_w_tiles = shard width in tiles, num_lines = lines with rows); only block_rows is
    chosen — the whole resident shard when it fits `budget` (the CB region below the resident shard buffers),
    sub-chunked otherwise."""
    from ttnn.operations._op_contract import UnsupportedAxisValue

    if shard is not None:
        k, cw_valid = shard.w_split, shard.core_w_tiles
        # Coarsest W-chunk whose one-tile-row footprint fits: the whole slice (no chunking) first. Chunking shrinks
        # the γ and RM staging terms only; the resident row (cb_x) is never chunked (x is read once).
        br_fit, cw, wc = 0, cw_valid, cw_valid
        for m in range(1, cw_valid + 1):
            wc = _div_up(cw_valid, m)
            if m > 1 and wc == _div_up(cw_valid, m - 1):
                continue
            cw = cw_valid if g.is_tile else _round_up(cw_valid, wc)  # RM: tilize chunks of CT width wc
            br_fit = (budget - _fixed_bytes(g, cw, wc)) // _per_row_bytes(g, cw, k)
            if br_fit >= 1:
                break
        if br_fit < 1:
            raise UnsupportedAxisValue("rms_norm: shard too wide for one tile-row of working CBs in L1")
        rows_unit = 1 if g.is_tile else TILE
        line_rows_max = max(_div_up(n, rows_unit) for (_, n) in shard.line_units)
        block_rows = max(1, min(line_rows_max, br_fit))
        regime = "LOCAL" if k == 1 else "W_SPLIT"
        xg = _xg_prescale(g, block_rows, cw, wc, k, budget)
        return Blocking(regime, k, cw, len(shard.lines), block_rows, wc, xg)

    k_fit = None
    for k in range(1, C + 1):
        if _footprint(g, 1, _div_up(Wt, k), k) <= L1_CB_BUDGET_BYTES:
            k_fit = k
            break
    if k_fit is None:
        raise UnsupportedAxisValue("rms_norm: row too wide to hold resident across the grid")
    k_occ = 1 if R >= C else max(1, min(C // R, Wt // MIN_CORE_W_TILES))
    k = max(k_fit, k_occ)
    cw = _div_up(Wt, k)
    w_split = _div_up(Wt, cw)  # every core of a line has >= 1 valid tile
    num_lines = min(R, C // w_split)
    line_rows_tiles_max = _div_up(R, num_lines)
    br_fit = (L1_CB_BUDGET_BYTES - _fixed_bytes(g, cw)) // _per_row_bytes(g, cw, w_split)
    block_rows = max(1, min(line_rows_tiles_max, br_fit))
    regime = "LOCAL" if w_split == 1 else "W_SPLIT"
    xg = _xg_prescale(g, block_rows, cw, cw, w_split, L1_CB_BUDGET_BYTES)
    return Blocking(regime, w_split, cw, num_lines, block_rows, cw, xg)


def _core_range_set(cores):
    """Merge a list of cores into per-row contiguous CoreRanges."""
    by_row = {}
    for c in cores:
        by_row.setdefault(c.y, []).append(c.x)
    ranges = []
    for y in sorted(by_row):
        xs = sorted(by_row[y])
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
                start = x
            prev = x
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index, num_pages, page_size, dtype, crs):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _is_sharded(t) -> bool:
    return t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED


def shard_placement(input_tensor, is_tile: bool, total_units: int, W: int) -> ShardPlacement:
    """Lines / positions / row ranges from the input's shard spec (op_design.md -> Sharded placements).

    `total_units` = tile-rows (TILE) or sticks (ROW_MAJOR) of the physical 2-D tensor. A shard-grid line whose rows
    are all padding (ragged grid) is dropped; a shard holding no valid W is refused (it cannot join a line's combine).
    """
    from ttnn.operations._op_contract import UnsupportedAxisValue

    mc = input_tensor.memory_config()
    ml = mc.memory_layout
    ss = mc.shard_spec
    if ss is None or mc.buffer_type != ttnn.BufferType.L1:
        raise UnsupportedAxisValue("rms_norm: sharded input must be an L1 shard with a shard spec")
    col_major = ss.orientation == ttnn.ShardOrientation.COL_MAJOR
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    if is_tile and (sh % TILE or sw % TILE):
        raise UnsupportedAxisValue("rms_norm: TILE shard extents must be tile multiples")
    sh_units = sh // TILE if is_tile else sh
    cores = ttnn.corerange_to_cores(ss.grid, row_wise=not col_major)

    if ml == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        grid_lines = [[c] for c in cores]  # shard i -> i-th core
    elif ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        grid_lines = [list(cores)]  # one line; shard j -> j-th core
    else:  # BLOCK: shard (row i, col j) -> core (x0 + j, y0 + i) [ROW_MAJOR] / (x0 + i, y0 + j) [COL_MAJOR]
        bb = ss.grid.bounding_box()
        x0, y0 = bb.start.x, bb.start.y
        by_line = {}
        for c in cores:
            line, pos = (c.x - x0, c.y - y0) if col_major else (c.y - y0, c.x - x0)
            by_line.setdefault(line, {})[pos] = c
        grid_lines = []
        for li in range(max(by_line) + 1):
            row = by_line.get(li, {})
            if sorted(row) != list(range(len(row))) or not row:
                raise UnsupportedAxisValue("rms_norm: BLOCK shard grid must be a rectangle")
            grid_lines.append([row[p] for p in range(len(row))])

    lines, line_units = [], []
    for li, line in enumerate(grid_lines):
        u0 = li * sh_units
        n = min(sh_units, total_units - u0)
        if n <= 0:
            continue  # ragged grid: an all-padding shard row
        for pos in range(len(line)):
            if pos * sw >= W:
                raise UnsupportedAxisValue("rms_norm: a shard holds no valid W (shard grid wider than the tensor)")
        lines.append(tuple(line))
        line_units.append((u0, n))
    k = len(lines[0])
    if any(len(l) != k for l in lines):
        raise UnsupportedAxisValue("rms_norm: shard-grid lines differ in length")
    return ShardPlacement(tuple(lines), tuple(line_units), sw, _div_up(sw, TILE), k, col_major)


@dataclass
class CoreWork:
    core: object
    pos: int  # position in its line (0 = root)
    root: object
    row_start: int  # first tile-row (global)
    line_rows: int  # tile-rows of this line
    stick_start: int  # RM: first stick (global)
    stick_end: int  # RM: one past the last valid stick
    w_start: int  # first W tile of the slice (global tile column)
    w_start_elems: int  # first W element of the slice
    w_valid: int  # tiles of the slice holding valid data
    w_valid_elems: int  # valid elements of the slice
    owns_partial_last: int  # slice holds the global last tile and W % 32 != 0


def create_program_descriptor(input_tensor, gamma, output_tensor, epsilon, compute_config):
    device = input_tensor.device()
    shape = [int(d) for d in input_tensor.shape]
    W, H = shape[-1], shape[-2]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    is_tile = input_tensor.layout == ttnn.TILE_LAYOUT
    dtype = input_tensor.dtype
    es = _element_bytes(input_tensor)
    has_gamma = gamma is not None
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT
    g_dtype = gamma.dtype if has_gamma else dtype
    gamma_is_block_float = has_gamma and g_dtype == ttnn.bfloat8_b
    x_sharded = _is_sharded(input_tensor)
    y_sharded = _is_sharded(output_tensor)

    # The ONE intermediate-format decision (numeric-formats §4): every intermediate CB (Σx² raw / partial / gathered,
    # rinv col / bcast / tile) follows the DEST width. Feeds the CB descriptors, the L1 budget (_per_row_bytes) and
    # the writer's gather / multicast byte counts.
    int_fmt = ttnn.float32 if compute_config.fp32_dest_acc_en else ttnn.bfloat16
    t_int = ttnn.tile_size(int_fmt)

    # ---- shape-derived extents (alignment-aware) ----
    Wt = _div_up(W, TILE)
    w_valid_last = W - TILE * (Wt - 1)
    if dtype == ttnn.bfloat8_b and (not is_tile or w_valid_last != TILE):
        # Block-float input: the reader's per-lane pad zeroing / RM stick paths address elements, which a bf8b tile
        # (shared-exponent section + mantissa section) does not have. feature_spec INVALID removes these cells.
        raise ValueError("rms_norm: bfloat8_b input requires TILE layout and a tile-aligned W")
    num_sticks = lead * H  # RM: total sticks (rows)
    R = lead * _div_up(H, TILE) if is_tile else _div_up(num_sticks, TILE)

    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    C = gx * gy

    # ---- placement: the shard grid fixes the cores of a sharded input ----
    shard = shard_placement(input_tensor, is_tile, R if is_tile else num_sticks, W) if x_sharded else None
    x_resident = x_sharded and is_tile  # TILE shard consumed in place (cb_x backed on it)
    y_resident = y_sharded and is_tile  # TILE output shard produced in place (cb_y backed on it)
    # A resident partial-W tile cannot be zeroed in place (it is the caller's tensor): its squares are packed as a
    # second per-row Σx² tile and masked by the collapse reduce's 0/1 lane mask (CB_WMASK).
    mask_partial = x_resident and w_valid_last < TILE

    # RM γ is read as a stick slice at byte offset w_start_elems·es_γ; RM shards put that offset on a 16 B (L1) grain,
    # which a DRAM read cannot start on -> stage an aligned window, then copy the slice core-locally.
    dram_align = int(ttnn._ttnn.device.get_dram_alignment())
    es_g = _element_bytes(gamma) if has_gamma else es
    gamma_rm_staged = bool(
        gamma_is_rm and shard is not None and (shard.shard_w * es_g) % dram_align != 0 and shard.w_split > 1
    )
    if has_gamma and not gamma_is_rm and shard is not None and shard.shard_w % TILE:
        raise ValueError("rms_norm: TILE γ needs tile-aligned W slices (ROW_MAJOR shards pair with ROW_MAJOR γ)")

    geo = OpGeometry(
        is_tile=is_tile,
        has_gamma=has_gamma,
        gamma_is_rm=gamma_is_rm,
        t_in=ttnn.tile_size(dtype),
        t_g=ttnn.tile_size(g_dtype),
        es_g=es_g,
        t_int=t_int,
        x_resident=x_resident,
        y_resident=y_resident,
        sumsq_parts=2 if mask_partial else 1,
        gamma_staged=gamma_rm_staged,
        dram_align=dram_align,
    )

    # CB region budget. Interleaved: the L1_CB_BUDGET_BYTES knob. Sharded: the CB region grows up from the L1
    # allocator base and must end below the resident shard buffers (allocated from the L1 top) — the real limit.
    budget = L1_CB_BUDGET_BYTES
    l1_addrs = [t.buffer_address() for t in (input_tensor, output_tensor) if _is_sharded(t)]
    if l1_addrs:
        l1_base = int(ttnn._ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1))
        budget = min(l1_addrs) - l1_base - SHARDED_CB_MARGIN_BYTES

    blk = select_blocking(geo, R, Wt, C, shard=shard, budget=budget)
    k, cw, br, wc = blk.w_split, blk.core_w_tiles, blk.block_rows, blk.w_chunk_tiles
    is_ws = blk.regime == "W_SPLIT"
    gamma_streams = has_gamma and blk.gamma_streams
    gd = _gamma_depth(cw, wc)

    # ---- per-core work: (line, position, row range, W slice) ----
    work = []
    if shard is None:
        cores = ttnn.grid_to_cores(blk.num_lines * k, gx, gy, row_wise=True)
        base_rows, extra_rows = divmod(R, blk.num_lines)
        row_start = 0
        for line in range(blk.num_lines):
            line_rows = base_rows + (1 if line < extra_rows else 0)
            line_cores = cores[line * k : (line + 1) * k]
            for pos, core in enumerate(line_cores):
                w_start = pos * cw
                work.append(
                    CoreWork(
                        core=core,
                        pos=pos,
                        root=line_cores[0],
                        row_start=row_start,
                        line_rows=line_rows,
                        stick_start=row_start * TILE,
                        stick_end=min((row_start + line_rows) * TILE, num_sticks),
                        w_start=w_start,
                        w_start_elems=w_start * TILE,
                        w_valid=min(cw, Wt - w_start),
                        w_valid_elems=min(cw * TILE, W - w_start * TILE),
                        owns_partial_last=0,
                    )
                )
            row_start += line_rows
    else:
        for line_cores, (u0, n) in zip(shard.lines, shard.line_units):
            if is_tile:
                row_start, line_rows, s0, s1 = u0, n, u0 * TILE, (u0 + n) * TILE
            else:
                row_start, line_rows, s0, s1 = u0 // TILE, _div_up(n, TILE), u0, u0 + n
            for pos, core in enumerate(line_cores):
                e0 = pos * shard.shard_w
                ev = min(shard.shard_w, W - e0)
                work.append(
                    CoreWork(
                        core=core,
                        pos=pos,
                        root=line_cores[0],
                        row_start=row_start,
                        line_rows=line_rows,
                        stick_start=s0,
                        stick_end=s1,
                        w_start=e0 // TILE,
                        w_start_elems=e0,
                        w_valid=_div_up(ev, TILE),
                        w_valid_elems=ev,
                        owns_partial_last=0,
                    )
                )
        cores = [w.core for w in work]
    for w in work:
        if is_tile and w_valid_last < TILE and w.w_start + w.w_valid == Wt:
            w.owns_partial_last = 1
    crs = _core_range_set(cores)

    # ---- CBs (identical descriptors on every participating core: uniform addresses) ----
    t_in = geo.t_in
    cbs = []
    if not is_tile:
        cbs.append(_cb(CB_X_RM, X_RM_DEPTH * wc, t_in, dtype, crs))
    if x_resident:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X, input_tensor, core_ranges=crs))
    else:
        cbs.append(_cb(CB_X, (X_BLOCK_DEPTH * br * cw) if is_tile else (br * cw), t_in, dtype, crs))
    if has_gamma:
        if gamma_is_rm:
            cbs.append(_cb(CB_GAMMA_RM, gd, wc * TILE * geo.es_g, g_dtype, crs))
        cbs.append(_cb(CB_GAMMA, gd * wc, geo.t_g, g_dtype, crs))
    if gamma_rm_staged:
        cbs.append(_cb(CB_GAMMA_STAGE, 1, _gamma_stage_bytes(geo, wc), g_dtype, crs))
    cbs.append(_cb(CB_SUMSQ_RAW, geo.sumsq_parts * br, t_int, int_fmt, crs))
    if mask_partial:
        cbs.append(_cb(CB_WMASK, 1, t_int, int_fmt, crs))
    collapse_rt = _collapse_reduce_tile(geo)
    if collapse_rt:
        cbs.append(_cb(CB_REDUCE_SCALER, 1, ttnn.tile_size(ttnn.bfloat16), ttnn.bfloat16, crs))
    norm_block = _norm_block(is_tile, y_resident, br, cw)
    if is_ws:
        cbs.append(_cb(CB_PARTIAL, br, t_int, int_fmt, crs))
        cbs.append(_cb(CB_GATHERED, k * br, t_int, int_fmt, crs))
        cbs.append(_cb(CB_RINV_BCAST, br, t_int, int_fmt, crs))
    if blk.xg_prescale:
        cbs.append(_cb(CB_XG, br * cw, t_int, int_fmt, crs))
    cbs.append(_cb(CB_RINV_COL, br, t_int, int_fmt, crs))
    cbs.append(_cb(CB_RINV, br, t_int, int_fmt, crs))
    if is_tile:
        if y_resident:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_Y, output_tensor, core_ranges=crs))
        else:
            cbs.append(_cb(CB_Y, Y_DEPTH * Y_CHUNK_TILES, t_in, dtype, crs))
    else:
        cbs.append(_cb(CB_OUT_RM, OUT_RM_DEPTH * wc, t_in, dtype, crs))

    semaphores = []
    if is_ws:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_PARTIALS, core_ranges=crs, initial_value=0))
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_RINV_LOCAL, core_ranges=crs, initial_value=0))

    # ---- runtime args ----
    x_page = input_tensor.buffer_page_size()
    y_page = output_tensor.buffer_page_size()
    g_page = gamma.buffer_page_size() if has_gamma else 0
    x_addr = input_tensor.buffer_address()
    y_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0

    reader_rt_by_core = {}
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    inv_w_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    for w in work:
        core = w.core
        root_v = device.worker_core_from_logical_core(w.root)
        is_root = 1 if w.pos == 0 else 0
        # byte offset of the slice inside one interleaved stick (sharded sticks hold only the slice)
        x_col_off = 0 if x_sharded else w.w_start_elems * es
        y_col_off = 0 if y_sharded else w.w_start_elems * es
        reader_rt_by_core[(core.x, core.y)] = [
            x_addr,
            w.row_start,
            w.line_rows,
            w.w_start,
            w.w_valid,
            w.w_valid_elems,
            w.owns_partial_last,
            is_root,
            w.stick_start,
            w.stick_end,
            x_col_off,
        ]
        writer_rt[core.x][core.y] = [
            y_addr,
            g_addr,
            w.row_start,
            w.line_rows,
            w.w_start,
            w.w_valid,
            w.w_valid_elems,
            is_root,
            w.pos,
            root_v.x,
            root_v.y,
            w.stick_start,
            w.stick_end,
            w.w_start_elems * geo.es_g,  # RM γ byte offset of the slice
            y_col_off,
        ]
        compute_rt[core.x][core.y] = [
            w.line_rows,
            is_root,
            inv_w_bits,
            eps_bits,
            w.w_valid - w.owns_partial_last,  # resident: fully valid tiles of the slice
            w.owns_partial_last,
        ]

    # ---- kernels ----
    reader_ct = [
        int(not is_tile),  # 0 is_rm
        int(is_ws),  # 1 is_wsplit
        br,  # 2 block_rows
        cw,  # 3 core_w_tiles
        READ_CHUNK_TILES,  # 4
        X_BLOCK_DEPTH,  # 5
        X_RM_DEPTH,  # 6
        t_in,  # 7 tile bytes of cb_x / cb_x_rm pages
        es,  # 8 input element bytes
        w_valid_last,  # 9
        x_page,  # 10 accessor page size (sharded RM: shard stick stride)
        Wt,  # 11
        k,  # 12 w_split
        SEM_PARTIALS,  # 13
        SEM_RINV_LOCAL,  # 14
        int(x_sharded),  # 15 input is an L1 shard (TILE: resident cb_x; RM: core-local re-stride)
        int(mask_partial),  # 16 owner of the partial-W tile fills cb_wmask
        wc,  # 17 w_chunk_tiles (RM staging chunk)
        int(collapse_rt),  # 18 fill the ReduceTile collapse's ones scaler (cb_reduce_scaler)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    def _reader(core_list):
        rt = ttnn.RuntimeArgs()
        for c in core_list:
            rt[c.x][c.y] = reader_rt_by_core[(c.x, c.y)]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
            core_ranges=_core_range_set(core_list),
            compile_time_args=list(reader_ct),
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

    # W_SPLIT: the line roots' readers never touch the multicast channel (the root is its sender; its own copy
    # is signalled by SEM_RINV_LOCAL), and the Mcast adapter requires every kernel attached on a sender core to
    # run on the channel's NoC — so the roots get their own reader descriptor with an absent channel.
    root_cores = [w.core for w in work if w.pos == 0] if is_ws else []
    member_cores = [w.core for w in work if w.pos != 0] if is_ws else list(cores)
    readers_absent = [_reader(root_cores)] if root_cores else []
    readers_attached = [_reader(member_cores)] if member_cores else []

    writer_ct = [
        int(not is_tile),  # 0 is_rm
        int(is_ws),  # 1 is_wsplit
        int(has_gamma),  # 2
        int(gamma_is_rm),  # 3
        br,  # 4 block_rows
        cw,  # 5 core_w_tiles
        Y_CHUNK_TILES,  # 6
        Y_DEPTH,  # 7
        t_in,  # 8 output tile bytes
        es,  # 9 output element bytes
        y_page,  # 10 output accessor page size (sharded RM: shard stick stride)
        g_page,  # 11 gamma accessor page size
        geo.es_g,  # 12 gamma element bytes
        geo.t_g,  # 13 gamma tile bytes
        Wt,  # 14
        k,  # 15 w_split
        SEM_PARTIALS,  # 16
        SEM_RINV_LOCAL,  # 17
        t_int,  # 18 intermediate tile bytes (gather / multicast)
        int(gamma_is_block_float),  # 19 TILE γ is block-float -> read whole tiles
        int(y_sharded),  # 20 output is an L1 shard (TILE: resident cb_y; RM: core-local re-stride)
        int(gamma_rm_staged),  # 21 RM γ slice read through an aligned staging window
        dram_align,  # 22
        wc,  # 23 w_chunk_tiles (RM staging / γ chunk)
        int(gamma_streams),  # 24 γ streamed per block in w_chunk_tiles chunks
        int(GATHER_COL_FACES),  # 25 L4: partial gather sends faces 0 + 2 only
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct = [
        int(not is_tile),  # 0 is_rm
        int(is_ws),  # 1 is_wsplit
        int(has_gamma),  # 2
        int(gamma_is_rm),  # 3
        br,  # 4 block_rows
        cw,  # 5 core_w_tiles
        k,  # 6 w_split
        int(bool(compute_config.math_approx_mode)),  # 7 rsqrt approx
        int(x_resident),  # 8 cb_x is the resident input shard (strided sumsq over valid tiles)
        int(mask_partial),  # 9 partial-W tile squared apart and masked in the collapse
        wc,  # 10 w_chunk_tiles (RM tilize / untilize chunk, γ chunk)
        int(gamma_streams),  # 11 γ streamed per block in w_chunk_tiles chunks
        int(blk.xg_prescale),  # 12 x·γ into cb_xg before the rinv wait
        norm_block,  # 13 DEST window of the normalize / prescale chains (NORM_DEST_BLOCK, ring-safe)
        int(collapse_rt),  # 14 collapse on the ReduceTile datapath (COLLAPSE_REDUCE_TILE)
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=crs,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )

    descriptor = ttnn.ProgramDescriptor(cbs=cbs, semaphores=semaphores)
    if is_ws:
        # rinv broadcast: one fixed in-group sender (the line root = position 0 of each group of w_split
        # consecutive cores in the line order), driven from the writer (NoC of WriterConfigDescriptor). Lines are
        # row-major groups (interleaved, ROW_MAJOR shards) or column-major groups (COL_MAJOR BLOCK/WIDTH shards).
        order = ttnn.McastCoreOrder.ColumnMajor if (shard is not None and shard.col_major) else ttnn.McastCoreOrder.RowMajor
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=RINV_MCAST_HANDSHAKE),
            crs,
            k,
            ttnn.McastFixedSenderConfig(sender_index=0),
            order,
        )
        mcast.attach(descriptor, MCAST_PREFIX, readers_attached + [writer], SEM_MCAST_FIRST)
    else:
        ttnn.attach_absent_mcast(writer, MCAST_PREFIX)
        for r in readers_attached:
            ttnn.attach_absent_mcast(r, MCAST_PREFIX)
    for r in readers_absent:
        ttnn.attach_absent_mcast(r, MCAST_PREFIX)
    descriptor.kernels = readers_absent + readers_attached + [writer, compute]
    return descriptor
