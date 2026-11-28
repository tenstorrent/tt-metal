# small_tensor_split: candidate work splits for g6_to_layout's interleaved regimes.
#
# Each variant is a host function (grid, geometry) -> Split. The bench monkeypatches the program
# descriptor's `_tile_split` (and `_choose_blocking` where the variant fixes the block) with it.
#
#   op kernels (host-only change, RT args (tile_begin, num_tiles)):
#     s1          the op's split_work_to_cores flattened tile range (baseline, unpatched)
#     nostraddle  each tile-row's columns cut into spans that never cross a tile-row (rows <= G), or
#                 whole tile-rows per core (rows > G)
#     rows        whole tile-rows per core (S2)
#     min<k>      at least k tiles per core: the flattened range over ceil(T / k) cores
#
#   experiment kernels (kernels_m<mode>/, R1 staged_reorder only; mode is compile-time):
#     ctl         s1's ranges, mode 0 (the op's code paths in the experiment copy): control
#     band<k>     mode 1: tiles cut into row bands of k rows; s1-like tile ranges per band
#     sub<j>      mode 1: s1's ranges; a one-tile-row block runs as sub-bands of j rows on its core
#                 (the writer drains sub-band k while the reader reorders k + 1)
#     fill[_s<j>] mode 1, the candidate rule: T < G -> each tile on floor(G / T) cores as near-even
#                 row bands (one (tile, band) per core); T >= G -> s1; plus sub-bands of j rows
#     rowflat     mode 2: flattened range of OUTPUT row segments (64 B of one row) over all cores
#     rowalign    mode 2, cut at row boundaries
from __future__ import annotations

from dataclasses import dataclass

import ttnn

TILE_HW = 32


def _div_up(a, b):
    return -(-a // b)


@dataclass
class Split:
    core_args: list  # [((x, y), args)]
    num_cores: int
    max_tiles: int  # largest number of (whole or partial) tiles a core touches (sizes the block)
    blocking: tuple = None  # (block_rows_tiles, block_cols_tiles) override, None: the op's rule
    kernel_mode: int = None  # None: the op's kernels; else kernels_m<mode>/ of this directory


def _cores(grid, n):
    gx, gy = grid
    assert n <= gx * gy
    return [(k % gx, k // gx) for k in range(n)]


def _even(total, parts):
    """sizes of `parts` near-equal pieces of `total` (larger first, as split_work_to_cores)."""
    q, r = divmod(total, parts)
    return [q + 1 if i < r else q for i in range(parts)]


def _flat(grid, geometry, n, *, band_args=None):
    """`n` cores, flattened tile range split near-evenly."""
    T = geometry.tensor_tiles
    n = min(n, T)
    sizes = _even(T, n)
    args, t = [], 0
    for core, size in zip(_cores(grid, n), sizes):
        args.append((core, (t, size) if band_args is None else (t, size, *band_args)))
        t += size
    return args, max(sizes)


def split_s1(grid, geometry):
    gx, gy = grid
    args, mx = _flat(grid, geometry, gx * gy)
    return Split(args, len(args), mx)


def split_min(k):
    def f(grid, geometry):
        gx, gy = grid
        n = min(gx * gy, _div_up(geometry.tensor_tiles, k))
        args, mx = _flat(grid, geometry, n)
        return Split(args, len(args), mx)

    return f


def split_rows(grid, geometry):
    gx, gy = grid
    rows, cols = geometry.tensor_rows_tiles, geometry.tensor_cols_tiles
    n = min(gx * gy, rows)
    args, g = [], 0
    for core, nrows in zip(_cores(grid, n), _even(rows, n)):
        args.append((core, (g * cols, nrows * cols)))
        g += nrows
    return Split(args, n, _div_up(rows, n) * cols)


def split_nostraddle(grid, geometry):
    gx, gy = grid
    G = gx * gy
    rows, cols = geometry.tensor_rows_tiles, geometry.tensor_cols_tiles
    if rows > G:
        return split_rows(grid, geometry)
    spans = max(1, G // rows)
    m = _div_up(cols, spans)
    spans = _div_up(cols, m)
    pieces = []
    for g in range(rows):
        c = 0
        for size in _even(cols, spans):
            pieces.append((g * cols + c, size))
            c += size
    args = [(core, piece) for core, piece in zip(_cores(grid, len(pieces)), pieces)]
    return Split(args, len(args), max(n for _, n in pieces))


def split_ctl(grid, geometry):
    """s1's ranges on the experiment kernels compiled in mode 0 (must time as s1)."""
    gx, gy = grid
    args, mx = _flat(grid, geometry, gx * gy)
    return Split(args, len(args), mx, None, 0)


def _rowflat_ranges(grid, geometry, n=None):
    gx, gy = grid
    units = geometry.out_pages * geometry.tensor_cols_tiles
    n = min(n or gx * gy, units)
    ranges, u = [], 0
    for size in _even(units, n):
        ranges.append((u, u + size))
        u += size
    return ranges


def _rowflat_split(grid, geometry, ranges, band_rows):
    cols = geometry.tensor_cols_tiles
    args = [(core, (u0, u1, band_rows)) for core, (u0, u1) in zip(_cores(grid, len(ranges)), ranges)]
    # block_cols must be the whole row in mode 2; one tile-row per block.
    return Split(args, len(args), cols, (1, cols), 2)


def split_rowflat(band_rows=TILE_HW):
    def f(grid, geometry):
        return _rowflat_split(grid, geometry, _rowflat_ranges(grid, geometry), band_rows)

    return f


def split_rowalign(grid, geometry):
    """Ranges cut at row boundaries only: 1/s of a row per core (s near-even spans of each row,
    out_rows * s <= G) when rows <= G, else whole rows near-evenly."""
    gx, gy = grid
    G = gx * gy
    out_rows, cols = geometry.out_pages, geometry.tensor_cols_tiles
    ranges = []
    if out_rows <= G:
        spans = max(1, min(cols, G // out_rows))
        for R in range(out_rows):
            c = 0
            for size in _even(cols, spans):
                ranges.append((R * cols + c, R * cols + c + size))
                c += size
    else:
        R = 0
        for nrows in _even(out_rows, G):
            ranges.append((R * cols, (R + nrows) * cols))
            R += nrows
    return _rowflat_split(grid, geometry, ranges, TILE_HW)


def _bands(live, b):
    """`b` near-even row bands of [0, live)."""
    out, lo = [], 0
    for size in _even(live, b):
        out.append((lo, lo + size))
        lo += size
    return out


def split_band(k, sub=TILE_HW):
    """Every tile-row cut into bands of k rows; the tile list is split over floor(G / bands) core
    groups, and each group's tile range runs once per band, on its own core (mode 1)."""

    def f(grid, geometry):
        gx, gy = grid
        G = gx * gy
        live = min(TILE_HW, geometry.height)
        bands = [(lo, min(lo + k, live)) for lo in range(0, live, k)]
        groups = min(max(1, G // len(bands)), geometry.tensor_tiles)
        pieces, t = [], 0
        for size in _even(geometry.tensor_tiles, groups):
            for lo, hi in bands:
                pieces.append((t, size, lo, hi, sub))
            t += size
        args = list(zip(_cores(grid, len(pieces)), pieces))
        return Split(args, len(args), max(p[1] for p in pieces), None, 1)

    return f


def _sub_only(grid, geometry, sub):
    """s1's tile ranges, each one-tile-row block cut into sub-bands of `sub` rows (mode 1)."""
    gx, gy = grid
    args, mx = _flat(grid, geometry, gx * gy)
    return Split([(core, (t, n, 0, TILE_HW, sub)) for core, (t, n) in args], len(args), mx, None, 1)


def split_fill(sub=TILE_HW):
    """The candidate rule. Fewer tiles than cores: every tile gets b = floor(G / T) near-even row
    bands of its live rows, one (tile, band) per core. Otherwise s1's tile ranges. Either way a
    one-tile-row block is cut into sub-bands of at most `sub` rows on its core (mode 1)."""

    def f(grid, geometry):
        gx, gy = grid
        G = gx * gy
        T = geometry.tensor_tiles
        if T < G:
            b = G // T
            pieces = []
            for t in range(T):
                g = t // geometry.tensor_cols_tiles
                live = min(TILE_HW, geometry.height - TILE_HW * (g % geometry.image_rows_tiles))
                for lo, hi in _bands(live, min(b, live)):
                    pieces.append((t, 1, lo, hi, sub))
            args = list(zip(_cores(grid, len(pieces)), pieces))
            return Split(args, len(args), 1, None, 1)
        args, mx = _flat(grid, geometry, G)
        return Split([(core, (t, n, 0, TILE_HW, sub)) for core, (t, n) in args], len(args), mx, None, 1)

    return f


def split_fill_lean(grid, geometry):
    """fill without sub-bands on the lean band walk (mode 3): T < G -> floor(G / T) row bands per
    tile, one (tile, band) per core; otherwise s1's ranges on the mode-0 kernels (the op's paths)."""
    gx, gy = grid
    G = gx * gy
    T = geometry.tensor_tiles
    live_max = min(TILE_HW, geometry.height)
    if T < G and min(G // T, live_max) >= 2:
        s = split_fill()(grid, geometry)
        return Split([(core, a[:4]) for core, a in s.core_args], s.num_cores, 1, None, 3)
    args, mx = _flat(grid, geometry, G)
    return Split(args, len(args), mx, None, 0)


def split_rect(b, m):
    """Rectangles of (b near-even row bands of the live rows) x (m-tile ranges): every m-tile
    group runs once per band, each (group, band) on its own core (lean band walk, mode 3)."""

    def f(grid, geometry):
        gx, gy = grid
        G = gx * gy
        T = geometry.tensor_tiles
        live = min(TILE_HW, geometry.height)
        bands = _bands(live, min(b, live))
        groups = _div_up(T, m)
        assert groups * len(bands) <= G, (b, m, T)
        pieces, t = [], 0
        for size in _even(T, groups):
            for lo, hi in bands:
                pieces.append((t, size, lo, hi))
            t += size
        args = list(zip(_cores(grid, len(pieces)), pieces))
        return Split(args, len(args), max(p[1] for p in pieces), None, 3)

    return f


# ---------------------------------------------------------------------------------------------
# The recommended rule.
# Measured on Blackhole p150b (11x10 grid), device kernel ns with the op's profiler zones:
#   ROW_NS        ~59 ns per output row a 1-tile core no longer moves (2 x 32-B reorder copies + 1
#                 row write): s1 1 tile x 32 rows vs one band of 5-6 rows on c2025.
#   stagger_ns(n) launch stagger (DEVICE KERNEL FIRST TO LAST START) of an n-core program:
#                 2 -> ~95, 4 -> ~190, 7-8 -> ~370, >= 16 -> 500-560 (flat).
#   BAND_PATH_NS  ~100 ns: the band read/reorder/write paths over the op's whole-tile paths.
ROW_NS = 59.0
BAND_PATH_NS = 100.0


def stagger_ns(n):
    return min(50.0 * max(0, n - 1), 550.0)


def band_count(geometry, G):
    """Row bands per tile for the staged_reorder band split (1 = the op's split)."""
    T = geometry.tensor_tiles
    live = min(TILE_HW, geometry.height)
    if T >= G or live < 2:
        return 1
    best_b, best_gain = 1, 0.0
    for b in range(2, min(G // T, live) + 1):
        rows = _div_up(live, b)
        gain = ROW_NS * (live - rows) - (stagger_ns(T * b) - stagger_ns(T)) - BAND_PATH_NS
        if gain > best_gain:
            best_b, best_gain = b, gain
    return best_b


def recommended_split(grid, geometry, regime):
    """Per-core work of an interleaved g6_to_layout call.

    staged_reorder (R1), fewer tiles than cores, band_count b >= 2: every tile on its own b cores,
        core k of a tile owning live rows [lo_k, hi_k) (near-even bands of the tile-row's live rows);
        runtime args (tile, 1, lo, hi) on the lean band walk (kernels_m3: BlockWalk + one band).
    packer_untilize (R3): ranges never cross a tile-row: rows <= G -> each tile-row's columns cut
        into floor(G / rows) near-even spans; rows > G -> whole tile-rows near-evenly.
    everything else: the op's split_work_to_cores flattened tile range (s1), unchanged kernels."""
    gx, gy = grid
    G = gx * gy
    if regime == "staged_reorder":
        b = band_count(geometry, G)
        if b >= 2:
            pieces = []
            for t in range(geometry.tensor_tiles):
                g = t // geometry.tensor_cols_tiles
                live_t = min(TILE_HW, geometry.height - TILE_HW * (g % geometry.image_rows_tiles))
                for lo, hi in _bands(live_t, min(b, live_t)):
                    pieces.append((t, 1, lo, hi))
            args = list(zip(_cores(grid, len(pieces)), pieces))
            return Split(args, len(args), 1, None, 3)
    if regime == "packer_untilize":
        return split_nostraddle(grid, geometry)
    return split_s1(grid, geometry)


# The bench sets the regime of the cell before it plans with "rec".
CURRENT = {"regime": None}


def lookup(name):
    """VARIANTS[name], plus rect<b>x<m> generated on demand and "rec" (the recommended rule)."""
    if name == "rec":
        return lambda grid, geometry: recommended_split(grid, geometry, CURRENT["regime"])
    if name.startswith("rect"):
        b, m = name[4:].split("x")
        return split_rect(int(b), int(m))
    return VARIANTS[name]


def _nz(split):
    """The same split on the zone-free copy of its kernel directory."""
    split.kernel_mode = f"{split.kernel_mode}_nz"
    return split


VARIANTS = {
    "s1": None,  # the op as it is
    "nostraddle": split_nostraddle,
    "rows": split_rows,
    "min2": split_min(2),
    "min4": split_min(4),
    "ctl": split_ctl,
    "rowflat": split_rowflat(),
    "rowalign": split_rowalign,
    "band16": split_band(16),
    "band8": split_band(8),
    "sub16": (lambda grid, geometry: _sub_only(grid, geometry, 16)),
    "sub8": (lambda grid, geometry: _sub_only(grid, geometry, 8)),
    "fill": split_fill(),
    "fill_lean": split_fill_lean,
    "fill_lean_nz": lambda grid, geometry: _nz(split_fill_lean(grid, geometry)),
    "ctl_nz": lambda grid, geometry: _nz(split_ctl(grid, geometry)),
    "fill_s16": split_fill(16),
    "fill_s8": split_fill(8),
}
