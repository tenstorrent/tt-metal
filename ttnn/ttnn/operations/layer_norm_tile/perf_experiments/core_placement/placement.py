# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core -> row-range placement variants for layer_norm_tile's `_split_rows`.

Every variant keeps the op's contract: (all_cores, ((core, row_tile_start, core_row_tiles), ...), max_rows).
Row ranges stay contiguous per core and are handed out in core-list order, so only WHICH core gets
which share (and which cores are used when rows < cores) changes.
Selected with env LN_PLACEMENT (one variant per pytest process: generic_op caches programs).
"""

import os

import ttnn

from ttnn.operations.layer_norm_tile import layer_norm_tile_program_descriptor as _pd


def _row_major_cores(grid):
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]


def _to_range_set(cores):
    """Merge per-row contiguous x runs into CoreRanges."""
    by_row = {}
    for core in cores:
        by_row.setdefault(core.y, []).append(core.x)
    ranges = []
    for y, xs in sorted(by_row.items()):
        xs = sorted(xs)
        run_start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(run_start, y), ttnn.CoreCoord(prev, y)))
            if x is not None:
                run_start = prev = x
    return ttnn.CoreRangeSet(ranges)


def _shares(num_rows, num_cores):
    used = min(num_rows, num_cores)
    base, extra = divmod(num_rows, used)
    return used, base, extra  # `extra` cores carry base+1


def _assign(ordered_cores, big_cores, num_rows):
    """ordered_cores: the used cores in row-range order; big_cores: subset carrying base+1."""
    used, base, _ = _shares(num_rows, len(ordered_cores))
    assignment = []
    start = 0
    big = {(c.x, c.y) for c in big_cores}
    for core in ordered_cores:
        n = base + 1 if (core.x, core.y) in big else base
        assignment.append((core, start, n))
        start += n
    assert start == num_rows, (start, num_rows)
    return _to_range_set(ordered_cores), tuple(assignment), max(a[2] for a in assignment)


def split_reversed(device, num_rows):
    """Reverse row-major order: the big group and (rows < cores) the used set sit at the highest y."""
    cores = list(reversed(_row_major_cores(device.compute_with_storage_grid_size())))
    used, _, extra = _shares(num_rows, len(cores))
    cores = cores[:used]
    return _assign(cores, cores[:extra], num_rows)


def split_interleave(device, num_rows):
    """Row-major order; big-group cores spread evenly across the grid (Bresenham)."""
    cores = _row_major_cores(device.compute_with_storage_grid_size())
    used, _, extra = _shares(num_rows, len(cores))
    cores = cores[:used]
    big = [cores[(i * used) // extra] for i in range(extra)] if extra else []
    return _assign(cores, big, num_rows)


def split_rev_interleave(device, num_rows):
    """Used set from the high-y end (reversed), big group spread evenly within it."""
    cores = list(reversed(_row_major_cores(device.compute_with_storage_grid_size())))
    used, _, extra = _shares(num_rows, len(cores))
    cores = cores[:used]
    big = [cores[(i * used) // extra] for i in range(extra)] if extra else []
    return _assign(cores, big, num_rows)


def split_rank(device, num_rows):
    """Explicit core ranking from env LN_RANK='x,y;x,y;...' (logical coords, fastest first); the
    first `extra` cores carry the big share, the first `used` cores are used; row ranges follow
    row-major order of the used set."""
    grid = device.compute_with_storage_grid_size()
    ranked = [ttnn.CoreCoord(*map(int, p.split(","))) for p in os.environ["LN_RANK"].split(";")]
    seen = {(c.x, c.y) for c in ranked}
    ranked += [c for c in _row_major_cores(grid) if (c.x, c.y) not in seen]
    used, _, extra = _shares(num_rows, len(ranked))
    chosen = ranked[:used]
    big = chosen[:extra]
    ordered = sorted(chosen, key=lambda c: (c.y, c.x))
    return _assign(ordered, big, num_rows)


def _ranked_split(device, num_rows, key):
    """Rank cores by `key` (best first): the first `extra` carry the big share; with rows < cores the
    first `used` are used (env LN_MAX_CORES caps the used count). Row ranges follow row-major order."""
    grid = device.compute_with_storage_grid_size()
    ranked = sorted(_row_major_cores(grid), key=lambda c: key(c, grid))
    max_cores = int(os.environ.get("LN_MAX_CORES", len(ranked)))
    used, _, extra = _shares(num_rows, min(len(ranked), max_cores))
    chosen = ranked[:used]
    ordered = sorted(chosen, key=lambda c: (c.y, c.x))
    return _assign(ordered, chosen[:extra], num_rows)


def _is_east(core, grid):
    # Logical x 0..6 are the west block (profiler x 1..7), 7..10 the east block (profiler x 11..14).
    return core.x >= 7


def split_west_hi(device, num_rows):
    """Big group (and the used set) on the west block, highest y first."""
    return _ranked_split(device, num_rows, lambda c, g: (_is_east(c, g), -c.y, c.x))


def split_hi_west(device, num_rows):
    """Highest y first, west before east within a row."""
    return _ranked_split(device, num_rows, lambda c, g: (-c.y, _is_east(c, g), c.x))


def _east_start(device, grid):
    """First logical x past the widest gap in the translated worker columns (the column block on the
    far side of the DRAM/harvest gap); grid.x when the columns are contiguous."""
    tx = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    jumps = [(tx[i] - tx[i - 1], i) for i in range(1, grid.x)]
    widest, at = max(jumps, default=(1, grid.x))
    return at if widest > 1 else grid.x


def split_hybrid(device, num_rows):
    """Used set (rows < cores): highest y first, west block before east within a row.
    Big group (rows > cores): west block, highest y first. Row ranges in row-major order."""
    grid = device.compute_with_storage_grid_size()
    east = _east_start(device, grid)
    cores = _row_major_cores(grid)
    used, _, extra = _shares(num_rows, len(cores))
    chosen = sorted(cores, key=lambda c: (-c.y, c.x >= east, c.x))[:used]
    big = sorted(chosen, key=lambda c: (c.x >= east, -c.y, c.x))[:extra]
    ordered = sorted(chosen, key=lambda c: (c.y, c.x))
    return _assign(ordered, big, num_rows)


VARIANTS = {
    "hybrid": split_hybrid,
    "west_hi": split_west_hi,
    "hi_west": split_hi_west,
    "baseline": None,
    "reversed": split_reversed,
    "interleave": split_interleave,
    "rev_interleave": split_rev_interleave,
    "rank": split_rank,
}


def _graduated(device, num_rows):
    import graduated_descriptor

    return graduated_descriptor._split_rows(device, num_rows)


VARIANTS["graduated"] = _graduated


def install():
    name = os.environ.get("LN_PLACEMENT", "baseline")
    fn = VARIANTS[name]
    if fn is not None:
        _pd._split_rows = fn
    return name
