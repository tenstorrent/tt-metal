# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor for the `interleaved_row_split` regime (op_design.md).

Work = the R x C grid of OUTPUT tiles. Tile-rows are split across the whole
compute grid (`split_work_to_cores(..., row_wise=True)`); every core owns the
rectangle (row_start_tile, core_rows_tiles, col_start_tile, core_cols_tiles).
Inside a core the rectangle is walked as blocks of
block_height_tiles x block_width_tiles; every block knob is defined once here
and all CB page counts / trip counts derive from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names) ---------------------------------------------
CB_RM_STICKS = 0  # reader -> compute: RM stick segments, tile-sized pages
CB_RM_STICKS_TAIL = 1  # same, for the ragged last column block (own ring quantum)
CB_STICK_STAGING = 2  # reader-private scratch: bank-grouped sticks (coalesced) / input tiles (retile)
CB_PAD_FILL = 3  # reader-private: one stick segment of the pad value (source of local fill copies)
CB_COPY_JOB = 4  # reader -> writer: split-reorder copy job per tile-row {dst, slot base, first stick}
CB_COPY_DONE = 5  # writer -> reader: split-reorder ack per tile-row (token page)
CB_TILES_OUT = 16  # compute -> writer: tiled output pages
CB_TILES_OUT_TAIL = 17  # same, for the ragged last column block

# --- Block knobs (single source of truth) -----------------------------------
IN_DEPTH_TILE_ROWS = 2  # cb_rm_sticks window, in tile-rows (double buffer)
OUT_DEPTH_TILE_ROWS = 2  # cb_tiles_out window, in tile-rows (double buffer)
L1_BLOCK_BUDGET_BYTES = 512 * 1024  # per-core budget for one column block's CB windows
FAST_TILIZE_MAX_WIDTH_TILES = 255  # fast-tilize LLK legality: block width < 256
LOW_L1_BLOCK_WIDTH_TILES = 8  # low_l1: dimension-independent block width
# Bank-coalesced reads: one wide read per DRAM bank per tile-row (whole-stick blocks only).
COALESCE_BANK_READS = True
# Coalescing trades 32 small DRAM reads for num_banks wide reads + 32 local copies per tile-row;
# it pays under grid-wide DRAM contention (BH: 110 cores 2x faster) and loses ~0.15us on a
# handful of cores (BH: 4-8 cores). Enable only when cores_used >= this * num_banks.
COALESCE_MIN_CORES_PER_BANK = 2
# Rotate each core's bank walk so the grid's requests spread over all banks at once.
BANK_STAGGER = True
# Segment (non-coalesced) reader: rotate each core's start stick within a tile-row (k % tile_h).
SEGMENT_BANK_STAGGER = True
# Coalesced reader staging ring: STAGING_SLOT_BUDGET_BYTES per slot sets staging_tile_rows per
# chunk (>= 1 whole tile-row); STAGING_SLOTS chunks are in flight at once (one trid each, <= 15).
STAGING_SLOT_BUDGET_BYTES = 16 * 1024
STAGING_SLOTS = 2  # >= 2: chunks in flight ahead of the one being reordered = STAGING_SLOTS - 1
# Rows of the FIRST chunk (None = staging_tile_rows): a short first chunk lands sooner, so the
# reorder/compute/write pipeline starts earlier under grid-wide DRAM contention.
FIRST_CHUNK_TILE_ROWS = None
# Split reorder (coalesced reader): fraction of every tile-row's stick copies the WRITER RISC does
# on its own NoC, in parallel with the reader's (0 = reader does all). Only with an interleaved
# (non-resident) output, single-packet sticks.
WRITER_COPY_SHARE = 0.0
COPY_JOB_PAGE_BYTES = 16  # one L1-aligned page: 3 x uint32 job words
# block_height_tiles: None => the core's whole row assignment (Phase 0: one row block).
BLOCK_HEIGHT_TILES = None
# interleaved_2d_split: also cut the columns across cores when the row axis cannot fill the grid.
ENABLE_2D_SPLIT = True
# Leading (batch) dims the reader can pad independently; rank <= 2 + this.
MAX_LEADING_DIMS = 4
# Sharded regimes (op_design.md): pin the work split to a side's resident shards when that side's
# grid holds at least this fraction of min(R, grid) cores; else fall back to the full-grid split.
PIN_OCCUPANCY_RATIO = 0.5
# Fill-row source offset: src = fill_base + (dst % FILL_PHASE_BYTES) keeps every local fill copy
# NoC-alignment congruent (any alignment <= this) and in element phase.
FILL_PHASE_BYTES = 64
# retile_tile_input: staging ring depth for the input tiles of one unit (lcm(in_tile_h, tile_h) rows
# x one column block). >= 2 overlaps the next unit's tile reads with this unit's de-face copies.
RETILE_STAGING_SLOTS = 2

TILE_WIDTH = 32


def _align_down(x, unit):
    return (x // unit) * unit


def compute_blocking(
    *, core_cols_tiles, in_page_bytes, out_page_bytes, in_elem_bytes, low_l1, read_alignment, staging_col_bytes=0
):
    """Column blocking: (block_width_tiles, num_col_blocks, last_block_width_tiles).

    `staging_col_bytes`: extra reader-private bytes per column tile (retile staging ring)."""
    per_col_tile_bytes = IN_DEPTH_TILE_ROWS * in_page_bytes + OUT_DEPTH_TILE_ROWS * out_page_bytes + staging_col_bytes
    if low_l1:
        max_bw = LOW_L1_BLOCK_WIDTH_TILES
    else:
        max_bw = min(FAST_TILIZE_MAX_WIDTH_TILES, L1_BLOCK_BUDGET_BYTES // per_col_tile_bytes)
    # Every stick-segment offset (col_start * 32 * elem) must stay read-aligned.
    seg_unit_bytes = TILE_WIDTH * in_elem_bytes
    align_unit_tiles = read_alignment // math.gcd(read_alignment, seg_unit_bytes)
    max_bw = max(align_unit_tiles, _align_down(max_bw, align_unit_tiles))

    block_width_tiles = min(core_cols_tiles, max_bw)
    num_col_blocks = math.ceil(core_cols_tiles / block_width_tiles)
    last_block_width_tiles = core_cols_tiles - (num_col_blocks - 1) * block_width_tiles
    return block_width_tiles, num_col_blocks, last_block_width_tiles


def _split_extent(total, parts, unit=1):
    """Contiguous partition of `total` into `parts` ranges, boundaries on multiples of `unit`.

    Returns [(start, count)] with counts ceil/floor over units (the remainder spread over the
    leading ranges, as split_work_to_cores does); the last range absorbs a partial unit.
    """
    units = math.ceil(total / unit)
    base, extra = divmod(units, parts)
    ranges, start = [], 0
    for k in range(parts):
        n_units = base + (1 if k < extra else 0)
        count = min(n_units * unit, total - start)
        ranges.append((start, count))
        start += count
    assert start == total
    return ranges


def _choose_split(num_cores, rows, col_units):
    """(cores_r, cores_c): the regime selector of op_design.md.

    interleaved_row_split (cores_c = 1, cores_r = min(R, grid)) unless the row axis cannot fill the
    grid (R < num_cores); then interleaved_2d_split picks cores_r x cores_c <= grid minimizing the
    busiest core's tile count, ties -> more row cores (longer stick segments), then fewer cores.
    """
    best = (math.ceil(rows / min(rows, num_cores)) * col_units, -min(rows, num_cores), 1)
    best_rc = (min(rows, num_cores), 1)
    if rows >= num_cores or not ENABLE_2D_SPLIT:
        return best_rc
    for cores_r in range(1, rows + 1):
        for cores_c in range(1, min(col_units, num_cores // cores_r) + 1):
            key = (math.ceil(rows / cores_r) * math.ceil(col_units / cores_c), -cores_r, cores_c)
            if key < best:
                best, best_rc = key, (cores_r, cores_c)
    return best_rc


def _rect_assignment(device, tensor_rows_tiles, tensor_cols_tiles, col_unit_tiles):
    """Per-core rectangles: [(core, row_start, rows, col_start, cols)], row-wise core order."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    col_units = math.ceil(tensor_cols_tiles / col_unit_tiles)
    cores_r, cores_c = _choose_split(num_cores, tensor_rows_tiles, col_units)
    # The first cores_r * cores_c cores of the grid, row-wise (one unit each).
    _, all_cores, _, _, _, _ = ttnn.split_work_to_cores(grid, cores_r * cores_c, row_wise=True)
    cores = ttnn.corerange_to_cores(all_cores, None, True)
    row_ranges = _split_extent(tensor_rows_tiles, cores_r)
    col_ranges = _split_extent(tensor_cols_tiles, cores_c, col_unit_tiles)
    assignment = []
    for ri, (row_start, rows) in enumerate(row_ranges):
        for ci, (col_start, cols) in enumerate(col_ranges):
            assignment.append((cores[ri * cores_c + ci], row_start, rows, col_start, cols))
    return all_cores, assignment, cores_c


def _flat_2d(shape):
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    return [rows, int(shape[-1])]


def _shard_rects(tensor, shape, tile_h, *, max_cols_tiles):
    """Per-core OUTPUT-tile rectangles of `tensor`'s resident L1 shards, or None.

    Returns (rects, shard_rows_tiles, shard_cols_tiles) with rects = [(core, row_start, rows,
    col_start, cols)] when every shard is ONE tile-aligned rectangle of the 2-D flattened tensor
    (rows = all dims but the last), sits alone on its core and spans its full shard width (so the
    shard is exactly `rows` tile-rows of `shard_cols_tiles` tiles, consumable/producible in place).
    A ragged last shard along the rows is fine (shorter runtime extent). Anything else (DRAM shards,
    several shards per core, a ragged last shard along the width, a non-rectangular ND box, a shard
    wider than one block) returns None and the op reads/writes that side through a TensorAccessor.
    """
    mem = tensor.memory_config()
    if mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED or mem.buffer_type != ttnn.BufferType.L1:
        return None
    nd = mem.nd_shard_spec
    if nd is None or len(shape) < 2 or int(shape[-2]) % tile_h != 0:
        return None
    s = [int(d) for d in nd.shard_shape]
    t = [int(d) for d in shape]
    if len(s) == 2:  # legacy 2-D spec: the tensor is viewed flattened to 2-D
        t = _flat_2d(t)
    elif len(s) < len(t):
        s = [1] * (len(t) - len(s)) + s
    elif len(s) > len(t):
        t = [1] * (len(s) - len(t)) + t
    n = len(t)
    if s[-1] % TILE_WIDTH != 0 or s[-1] // TILE_WIDTH > max_cols_tiles or t[-1] % s[-1] != 0:
        return None
    g = [math.ceil(td / sd) for td, sd in zip(t, s)]
    num_shards = math.prod(g)
    cores_in_grid = nd.grid.num_cores()
    if num_shards > cores_in_grid:
        return None
    strategy = nd.shard_distribution_strategy
    row_major = nd.orientation == ttnn.ShardOrientation.ROW_MAJOR
    if strategy == ttnn.ShardDistributionStrategy.GRID_2D:
        ranges = nd.grid.ranges()
        if n != 2 or len(ranges) != 1:
            return None
        origin = ranges[0].start
        core_of = lambda idx: (
            ttnn.CoreCoord(origin.x + idx[1], origin.y + idx[0])
            if row_major
            else ttnn.CoreCoord(origin.x + idx[0], origin.y + idx[1])
        )
    elif strategy == ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D:
        cores = ttnn.corerange_to_cores(nd.grid, None, row_major)
        core_of = lambda idx: cores[_linear(idx, g)]
    else:
        return None

    rects = []
    for i in range(num_shards):
        idx = _delinear(i, g)
        start = [a * b for a, b in zip(idx, s)]
        ext = [min(sd, td - st) for sd, td, st in zip(s, t, start)]
        # Rows (dims 0..n-2) must form one contiguous run of the flattened row index.
        k = max((d for d in range(n - 1) if ext[d] < t[d]), default=0)
        if any(ext[j] != 1 for j in range(k)):
            return None
        row0 = _linear(start[: n - 1], t[: n - 1])
        nrows = math.prod(ext[k : n - 1])
        if row0 % tile_h != 0 or nrows % tile_h != 0:
            return None
        rects.append((core_of(idx), row0 // tile_h, nrows // tile_h, start[-1] // TILE_WIDTH, ext[-1] // TILE_WIDTH))
    shard_rows = math.prod(s[:-1])
    if shard_rows % tile_h != 0:
        return None
    return rects, shard_rows // tile_h, s[-1] // TILE_WIDTH


def _linear(idx, dims):
    v = 0
    for i, d in zip(idx, dims):
        v = v * d + i
    return v


def _delinear(v, dims):
    idx = []
    for d in reversed(dims):
        idx.append(v % d)
        v //= d
    return list(reversed(idx))


def _same_rects(a, b):
    key = lambda r: (r[0].x, r[0].y, r[1], r[2], r[3], r[4])
    return sorted(map(key, a)) == sorted(map(key, b))


def _resident_cb(index, tensor, page_bytes, num_pages, cores, tile):
    """CB backed on `tensor`'s resident shard (zero-copy), re-paged to tile-sized pages."""
    desc = ttnn.cb_descriptor_from_sharded_tensor(index, tensor, total_size=num_pages * page_bytes, core_ranges=cores)
    desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=index, data_format=tensor.dtype, page_size=page_bytes, tile=ttnn.TileDescriptor(tile)
        )
    ]
    return desc


def _cb(index, num_pages, page_bytes, dtype, cores, tile=None):
    fmt_kwargs = dict(buffer_index=index, data_format=dtype, page_size=page_bytes)
    if tile is not None:
        fmt_kwargs["tile"] = ttnn.TileDescriptor(tile)
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(**fmt_kwargs)],
    )


def create_program_descriptor(input_tensor, output_tensor, *, low_l1=False, padded_shape=None, fill_word=0):
    """`padded_shape`: the output's padded shape (rank >= 2), None = no padding. The input is
    viewed left-expanded with 1s to the same rank; everything past it is filled with `fill_word`
    (the fill encoded in the input format, replicated to 32 bits)."""
    device = input_tensor.device()
    in_shape = [int(d) for d in input_tensor.shape]

    tile_h, tile_w = output_tensor.tile.tile_shape
    assert tile_w == TILE_WIDTH

    # --- retile_tile_input: a TILE input at in_tile_h is de-faced by the reader into the same
    # row layout cb_rm_sticks carries on the ROW_MAJOR path (compute / writer unchanged).
    retile = input_tensor.layout == ttnn.TILE_LAYOUT
    if retile:
        assert padded_shape is None, "tilize: padding a TILE input is not supported"
        in_tile_h = int(input_tensor.tile.tile_shape[0])
        in_face_h = int(input_tensor.tile.face_shape[0])
        in_tile_page_bytes = input_tensor.buffer_aligned_page_size()
        # Input tile-rows per unit (one staging slot): a unit is lcm(in_tile_h, tile_h) rows; both
        # are powers of two, so it is max(in_tile_h, tile_h) rows.
        unit_in_tile_rows = max(1, tile_h // in_tile_h)
    else:
        in_tile_h = in_face_h = in_tile_page_bytes = 1
        unit_in_tile_rows = 1

    out_shape = list(padded_shape) if padded_shape is not None else list(in_shape)
    assert len(out_shape) >= 2 and len(out_shape) >= len(in_shape)
    in_view = [1] * (len(out_shape) - len(in_shape)) + in_shape
    has_padding = in_view != out_shape
    out_lead, in_lead = out_shape[:-2], in_view[:-2]
    assert len(out_lead) <= MAX_LEADING_DIMS, f"tilize: rank {len(out_shape)} > {MAX_LEADING_DIMS + 2}"
    out_lead = [1] * (MAX_LEADING_DIMS - len(out_lead)) + out_lead
    in_lead = [1] * (MAX_LEADING_DIMS - len(in_lead)) + in_lead
    leading_pad = out_lead != in_lead

    # --- Work geometry (output = padded shape; per-image ceil, then fold leading dims) --------
    in_image_height_sticks = in_view[-2]
    tile_rows_per_image = math.ceil(out_shape[-2] / tile_h)
    tensor_rows_tiles = math.prod(out_shape[:-2]) * tile_rows_per_image
    tensor_cols_tiles = math.ceil(out_shape[-1] / TILE_WIDTH)
    # retile: the input's own tile grid (per-image ceil at in_tile_h; W is tile-padded the same).
    in_tile_rows_per_image = math.ceil(in_view[-2] / in_tile_h) if retile else 0
    in_cols_tiles = math.ceil(in_view[-1] / TILE_WIDTH)

    # --- Page sizes ---------------------------------------------------------
    in_elem_bytes = input_tensor.element_size()
    in_page_bytes = tile_h * TILE_WIDTH * in_elem_bytes  # one tile's worth of RM data
    out_page_bytes = output_tensor.buffer_page_size()  # one output tile
    stick_page_bytes = input_tensor.buffer_aligned_page_size()
    in_row_bytes = in_view[-1] * in_elem_bytes  # logical data bytes of one input stick
    # A width/block-sharded RM input pages each stick into shard-width pieces: page id of
    # (stick, piece) = stick * in_pages_per_stick + piece (interleaved / height: one page per stick).
    in_page_data_bytes = input_tensor.buffer_page_size()
    in_pages_per_stick = max(1, math.ceil(in_row_bytes / in_page_data_bytes))

    # --- Per-core rectangles --------------------------------------------------
    read_alignment = (
        ttnn.get_dram_alignment()
        if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        else ttnn.get_l1_alignment()
    )
    # Every stick-segment offset (col_start * 32 * elem) must stay read-aligned.
    seg_unit_bytes = TILE_WIDTH * in_elem_bytes
    align_unit_tiles = read_alignment // math.gcd(read_alignment, seg_unit_bytes)

    retile_staging_col_bytes = RETILE_STAGING_SLOTS * unit_in_tile_rows * in_tile_page_bytes if retile else 0

    def blocking_for(core_cols_tiles):
        return compute_blocking(
            core_cols_tiles=core_cols_tiles,
            in_page_bytes=in_page_bytes,
            out_page_bytes=out_page_bytes,
            in_elem_bytes=in_elem_bytes,
            low_l1=low_l1,
            read_alignment=read_alignment,
            staging_col_bytes=retile_staging_col_bytes,
        )

    # --- Regime selector (op_design.md "Selection function") ---------------------------------
    # A side whose L1 shards are natively consumable/producible pins the work split to its shards
    # (one block = the whole shard); the other side goes through a TensorAccessor (interleaved, or
    # sharded non-local). Input shards are resident only without padding (the input is the output's
    # exact prefix) and when a shard row is exactly the block's stick segment (no stride padding).
    max_block_tiles = blocking_for(FAST_TILIZE_MAX_WIDTH_TILES * align_unit_tiles)[0]
    in_shards = None
    if retile:
        # A TILE shard is the reader's de-face source in place (no NoC read); it must hold whole
        # units (lcm(in_tile_h, tile_h) rows) so every unit's input tiles are local.
        unit_h = max(in_tile_h, tile_h)
        if _shard_rects(input_tensor, in_shape, unit_h, max_cols_tiles=max_block_tiles) is not None:
            in_shards = _shard_rects(input_tensor, in_shape, tile_h, max_cols_tiles=max_block_tiles)
    elif not has_padding:
        in_shards = _shard_rects(input_tensor, in_shape, tile_h, max_cols_tiles=max_block_tiles)
        if in_shards is not None and stick_page_bytes != in_shards[2] * seg_unit_bytes:
            in_shards = None
    out_shards = _shard_rects(output_tensor, out_shape, tile_h, max_cols_tiles=max_block_tiles)
    grid = device.compute_with_storage_grid_size()
    pin_min_cores = min(tensor_rows_tiles, grid.x * grid.y) * PIN_OCCUPANCY_RATIO
    in_resident = out_resident = False
    if in_shards and out_shards and _same_rects(in_shards[0], out_shards[0]):
        regime, in_resident, out_resident = "sharded_same_spec_in_place", True, True
    elif in_shards and len(in_shards[0]) >= pin_min_cores:
        regime, in_resident = "sharded_input_pinned", True
    elif out_shards and len(out_shards[0]) >= pin_min_cores:
        regime, out_resident = "sharded_output_pinned", True
    else:
        regime = "full_grid"  # interleaved_row_split / interleaved_2d_split / sharded_full_grid
    if regime == "full_grid":
        all_cores, assignment, cores_c = _rect_assignment(
            device, tensor_rows_tiles, tensor_cols_tiles, align_unit_tiles
        )
    else:
        assignment = (in_shards if in_resident else out_shards)[0]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(e[0], e[0]) for e in assignment])
        cores_c = len({e[3] for e in assignment})

    # Column groups: cores whose rectangles have the same column extent share one column walk
    # (CT block widths), so they share one kernel/CB set. At most two (cores_c > 1, ragged last).
    groups = {}  # core_cols_tiles -> list of assignment entries
    for entry in assignment:
        groups.setdefault(entry[4], []).append(entry)

    # Bank-coalesced reads need the block to be the whole stick (contiguous in its bank), an
    # interleaved source (round-robin page -> bank) and no padding (every stick fully present).
    in_mem = input_tensor.memory_config()
    num_banks = ttnn.get_memory_view(device, in_mem.buffer_type).num_banks
    _, whole_width_blocks, _ = blocking_for(tensor_cols_tiles)
    coalesce = (
        COALESCE_BANK_READS
        and not retile
        and cores_c == 1
        and whole_width_blocks == 1
        and not has_padding
        and in_mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and tile_h > 1
        and in_image_height_sticks % tile_h == 0
        and len(assignment) >= COALESCE_MIN_CORES_PER_BANK * num_banks
    )
    max_core_rows = max(entry[2] for entry in assignment)
    staging_tile_rows = max(1, min(max_core_rows, STAGING_SLOT_BUDGET_BYTES // (tile_h * stick_page_bytes)))
    # One bank group of a slot (the reader derives the same max_per_group in-kernel).
    staging_group_bytes = math.ceil(staging_tile_rows * tile_h / num_banks) * stick_page_bytes
    staging_slot_bytes = num_banks * staging_group_bytes
    first_chunk_tile_rows = min(staging_tile_rows, FIRST_CHUNK_TILE_ROWS or staging_tile_rows)
    staging_slots = max(2, STAGING_SLOTS)
    staging_bytes = staging_slots * staging_slot_bytes
    # Bank stagger (coalesced reader): core k starts its bank walk at bank k % groups, where
    # groups = min(num_banks, sticks in the smallest chunk) so the offset is always in range.
    min_chunk_sticks = min(min(entry[2] for entry in assignment), first_chunk_tile_rows) * tile_h
    stagger_groups = min(num_banks, min_chunk_sticks)
    core_index = {(entry[0].x, entry[0].y): k for k, entry in enumerate(assignment)}

    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    out_tile = ttnn.Tile([tile_h, TILE_WIDTH])
    kernels, cbs = [], []
    for core_cols_tiles, entries in groups.items():
        group_cores = (
            all_cores if len(groups) == 1 else ttnn.CoreRangeSet([ttnn.CoreRange(e[0], e[0]) for e in entries])
        )
        block_width_tiles, num_col_blocks, last_block_width_tiles = blocking_for(core_cols_tiles)
        has_tail = last_block_width_tiles != block_width_tiles

        # --- Circular buffers (per group: ring quanta = this group's block widths) ----------
        # Resident sides: the CB IS the shard (zero-copy, whole shard = one block).
        if in_resident and not retile:
            shard_pages = in_shards[1] * in_shards[2]
            cbs.append(_resident_cb(CB_RM_STICKS, input_tensor, in_page_bytes, shard_pages, group_cores, out_tile))
        else:
            cbs.append(
                _cb(
                    CB_RM_STICKS,
                    IN_DEPTH_TILE_ROWS * block_width_tiles,
                    in_page_bytes,
                    input_tensor.dtype,
                    group_cores,
                    out_tile,
                )
            )
        if out_resident:
            shard_pages = out_shards[1] * out_shards[2]
            cbs.append(_resident_cb(CB_TILES_OUT, output_tensor, out_page_bytes, shard_pages, group_cores, out_tile))
        else:
            cbs.append(
                _cb(
                    CB_TILES_OUT,
                    OUT_DEPTH_TILE_ROWS * block_width_tiles,
                    out_page_bytes,
                    output_tensor.dtype,
                    group_cores,
                    out_tile,
                )
            )
        if coalesce:
            cbs.append(_cb(CB_STICK_STAGING, 1, staging_bytes, input_tensor.dtype, group_cores))
        if retile and not in_resident:
            retile_slot_bytes = unit_in_tile_rows * block_width_tiles * in_tile_page_bytes
            cbs.append(
                _cb(CB_STICK_STAGING, 1, RETILE_STAGING_SLOTS * retile_slot_bytes, input_tensor.dtype, group_cores)
            )
        # Split reorder: the writer copies its share of each tile-row's sticks. Both kernels drop
        # the split (share -> 0) when a stick copy is wider than one NoC packet.
        copy_seg_bytes = block_width_tiles * seg_unit_bytes
        writer_copy_sticks = int(tile_h * WRITER_COPY_SHARE) if coalesce and not out_resident else 0
        if writer_copy_sticks:
            for cb_index in (CB_COPY_JOB, CB_COPY_DONE):
                cbs.append(_cb(cb_index, IN_DEPTH_TILE_ROWS, COPY_JOB_PAGE_BYTES, input_tensor.dtype, group_cores))
        fill_row_bytes = block_width_tiles * seg_unit_bytes + FILL_PHASE_BYTES
        if has_padding:
            # + FILL_PHASE_BYTES so the kernel can align the row base up to FILL_PHASE_BYTES.
            cbs.append(_cb(CB_PAD_FILL, 1, fill_row_bytes + FILL_PHASE_BYTES, input_tensor.dtype, group_cores))
        if has_tail:
            # The ragged last column block gets its own ring so every ring is filled in
            # one quantum (no contiguous block can straddle the ring end).
            cbs.append(
                _cb(
                    CB_RM_STICKS_TAIL,
                    IN_DEPTH_TILE_ROWS * last_block_width_tiles,
                    in_page_bytes,
                    input_tensor.dtype,
                    group_cores,
                    out_tile,
                )
            )
            cbs.append(
                _cb(
                    CB_TILES_OUT_TAIL,
                    OUT_DEPTH_TILE_ROWS * last_block_width_tiles,
                    out_page_bytes,
                    output_tensor.dtype,
                    group_cores,
                    out_tile,
                )
            )

        # --- Kernel args -----------------------------------------------------------------
        col_geometry = [block_width_tiles, last_block_width_tiles, num_col_blocks, int(has_tail)]
        reader_ct = [
            CB_RM_STICKS,
            CB_RM_STICKS_TAIL,
            tile_h,
            seg_unit_bytes,  # bytes of one tile-column of one stick
            in_image_height_sticks,
            tile_rows_per_image,
            stick_page_bytes,
            *col_geometry,
            int(coalesce),
            num_banks,
            CB_STICK_STAGING,
            staging_tile_rows,
            int(has_padding),
            CB_PAD_FILL,
            in_row_bytes,
            fill_row_bytes,
            FILL_PHASE_BYTES,
            int(leading_pad),
            *out_lead,
            *in_lead,
            int(in_resident),
            in_pages_per_stick,
            in_page_data_bytes,
            staging_slots,
            writer_copy_sticks,
            CB_COPY_JOB,
            CB_COPY_DONE,
            first_chunk_tile_rows,
            int(retile),
            in_tile_h,
            in_face_h,
            in_tile_page_bytes,
            in_tile_rows_per_image,
            in_cols_tiles,
            RETILE_STAGING_SLOTS,
            in_shards[2] if (retile and in_resident) else 0,  # resident shard width, tiles
        ]
        reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        writer_ct = [
            CB_TILES_OUT,
            CB_TILES_OUT_TAIL,
            tensor_cols_tiles,
            out_page_bytes,
            *col_geometry,
            int(out_resident),
            writer_copy_sticks,
            CB_COPY_JOB,
            CB_COPY_DONE,
            copy_seg_bytes,
            num_banks,
            staging_group_bytes,
            stick_page_bytes,
        ]
        writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        compute_ct = [CB_RM_STICKS, CB_RM_STICKS_TAIL, CB_TILES_OUT, CB_TILES_OUT_TAIL, *col_geometry]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, row_start_tile, core_rows_tiles, col_start_tile, _ in entries:
            block_height_tiles = (
                core_rows_tiles if BLOCK_HEIGHT_TILES is None else min(BLOCK_HEIGHT_TILES, core_rows_tiles)
            )
            rect = [row_start_tile, core_rows_tiles, col_start_tile, block_height_tiles]
            k = core_index[(core.x, core.y)]
            if coalesce:
                stagger = (k % stagger_groups) if BANK_STAGGER else 0
            else:  # segment reader: rotate the start stick of each tile-row
                stagger = (k % tile_h) if SEGMENT_BANK_STAGGER else 0
            reader_rt[core.x][core.y] = [in_addr, *rect, stagger, fill_word]
            writer_rt[core.x][core.y] = [out_addr, *rect]
            compute_rt[core.x][core.y] = [core_rows_tiles, block_height_tiles]

        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
                core_ranges=group_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
                core_ranges=group_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
                core_ranges=group_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
