# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm — realization of op_design.md's Blocking Model.

Every block factor, buffer depth and core-assignment named in the design is a
parameter here, defined exactly ONCE and derived from downstream:

    NUM_ROW_GROUPS        (g)  — independent row-group rectangles on the grid
    NUM_HIDDEN_SLICES     (s)  — cores splitting the reduced (hidden) axis inside a rect
    SLICE_HIDDEN_TILES    (S)  — the hidden block extent = the whole slice
    BLOCK_ROWS            (B)  — the row block extent (coarsest that fits L1)
    OUT_CB_DEPTH / RM_IN_DEPTH / RM_OUT_DEPTH — buffer-depth knobs
    HIDDEN_TILES_PER_CORE_FLOOR — hidden-granularity tuning knob
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
OUT_CB_DEPTH = 2  # cb_output_tiles double buffer (2.78x overlap, double_buffer/report.md)
RM_IN_DEPTH = 2  # cb_rm_stage_in  tile-row window
RM_OUT_DEPTH = 2  # cb_rm_stage_out tile-row window
IN_CB_DEPTH = 1  # cb_input_tiles: exactly one block resident (in-place rewrite needs wr==rd)
DM_CHUNK_TILES = 8  # tiles per NoC barrier, reader and writer alike (the ROW_MAJOR
# kernels convert this byte budget into sticks — see RM_CHUNK_STICKS there)

# IN_CB_DEPTH is load-bearing at 1, not a free knob: the two in-place rewrites of x
# (mask_tail_block, scale_block) rely on get_write_ptr() == get_read_ptr(), which only
# holds when cb_input_tiles' capacity is EXACTLY the live block.  Turning it to 2 would
# silently write x*r into the wrong half.  The overlap perf lamp must therefore be
# measured as "smaller block_rows + a second buffer", never "same block, deeper CB".
assert IN_CB_DEPTH == 1, "cb_input_tiles must be exactly one block deep (in-place rewrite)"

# L1 budget. `l1_size_per_core()` is not bound to Python; the fallback is a
# single named constant (conservative for WH/BH, both >= 1.4 MB usable).
L1_SIZE_PER_CORE_FALLBACK = 1024 * 1024
L1_RESERVE = 96 * 1024  # firmware / kernel text / semaphores headroom

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
        total_sticks = input_tensor.buffer_num_pages()
        row_tiles = _div_up(total_sticks, TILE_DIM)

    return row_tiles, hidden_tiles, w, total_sticks


# ---------------------------------------------------------------------------
# L1 fit predicate — mirrors l1_ledger.md's footprint_tiles() exactly
# ---------------------------------------------------------------------------


def _footprint_bytes(block_rows, slice_tiles, num_slices, *, is_row_major, has_gamma, bytes_):
    b, s_, s = block_rows, slice_tiles, num_slices
    total = b * s_ * bytes_["in_tile"] * IN_CB_DEPTH  # cb_input_tiles
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


def _footprint_bytes_sharded(block_rows, slice_tiles, num_slices, *, is_row_major, has_gamma, bytes_):
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
        total += b * s_ * bytes_["in_tile"] * IN_CB_DEPTH  # cb_input_tiles (tilize target)
        total += RM_IN_DEPTH * s_ * bytes_["in_tile"]  # cb_rm_stage_in
        total += RM_OUT_DEPTH * s_ * bytes_["out_tile"]  # cb_rm_stage_out
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
    block_rows = 1
    for cand in range(shard_rows, 0, -1):
        # A divisor keeps every block the same size, which is what lets the
        # resident-shard CB stay exactly full at every block boundary (the
        # in-place rewrite of x needs get_write_ptr() == get_read_ptr()).
        if shard_rows % cand:
            continue
        if (
            _footprint_bytes_sharded(
                cand, slice_tiles, num_hidden_slices, is_row_major=is_row_major, has_gamma=has_gamma, bytes_=bytes_
            )
            <= budget
        ):
            block_rows = cand
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
        "rect_w": 0,
        "rect_h": 0,
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
    slice_cap = min(hidden_tiles, max(1, _div_up(hidden_tiles, HIDDEN_TILES_PER_CORE_FLOOR)), gx * gy)

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

    # Coarsest row block that fits: default = the whole per-core assignment.
    base, rem = divmod(row_tiles, num_row_groups)
    max_rows = base + (1 if rem else 0)
    block_rows = max_rows
    while block_rows > 1 and _footprint_bytes(
        block_rows, slice_tiles, num_hidden_slices, is_row_major=is_row_major, has_gamma=has_gamma, bytes_=bytes_
    ) > _l1_working_budget(device):
        block_rows -= 1

    return {
        "grid": (gx, gy),
        "row_tiles": row_tiles,
        "hidden_tiles": hidden_tiles,
        "num_row_groups": num_row_groups,
        "num_hidden_slices": num_hidden_slices,
        "slice_hidden_tiles": slice_tiles,
        "block_rows": block_rows,
        "rect_w": rect_w,
        "rect_h": rect_h,
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
    rect_w, rect_h = plan["rect_w"], plan["rect_h"]
    sharded = plan["sharded"]
    shard_rows = plan["shard_rows"]
    shard_tiles = plan["shard_tiles"]
    shard_w = plan["shard_w"]

    shape = list(input_tensor.shape)
    W = shape[-1]
    in_elem = _elem_bytes(input_tensor)
    out_elem = _elem_bytes(output_tensor)
    gamma_elem = _elem_bytes(gamma) if has_gamma else in_elem
    total_sticks = input_tensor.buffer_num_pages() if is_row_major else 0

    mask_valid_w = W % TILE_DIM
    mask_enabled = (mask_valid_w != 0) and not is_row_major

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
    mcast_by_group = {}
    if S_COUNT > 1:
        cfg = ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=True,
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
            cbs.append(_cb(CB_INPUT_TILES, IN_CB_DEPTH * B * S, in_tile, input_tensor.dtype))
        else:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor))
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor))
    else:
        # capacity == live set exactly: the in-place rewrite of x needs
        # get_write_ptr == get_read_ptr, which only holds at a full CB.
        cbs.append(_cb(CB_INPUT_TILES, IN_CB_DEPTH * B * S, in_tile, input_tensor.dtype))
        if not is_row_major:
            cbs.append(_cb(CB_OUTPUT_TILES, OUT_CB_DEPTH * S, out_tile, output_tensor.dtype))

    if mask_enabled:
        # Only the W-mask path (TILE layout with W % 32 != 0) touches this CB;
        # both the reader's prepare_reduce_mask and the compute kernel's
        # mask_tail_block are gated on the same predicate.  Matches l1_ledger.md.
        cbs.append(_cb(CB_W_MASK, 1, bf16_tile, ttnn.bfloat16))
    if has_gamma:
        cbs.append(_cb(CB_GAMMA_TILES, S, gamma_tile, gamma.dtype))
    if S_COUNT > 1:
        cbs.append(_cb(CB_GATHERED_PARTIALS, S_COUNT * B, stat_tile, ttnn.float32))
        cbs.append(_cb(CB_RMS_BCAST, B, stat_tile, ttnn.float32))
    if is_row_major:
        cbs.append(_cb(CB_RM_STAGE_IN, RM_IN_DEPTH * S, in_tile, input_tensor.dtype))
        cbs.append(_cb(CB_RM_STAGE_OUT, RM_OUT_DEPTH * S, out_tile, output_tensor.dtype))

    # ---- compile-time args ----
    # A TILE shard is consumed in place, so cb_input_tiles' capacity is the WHOLE
    # resident shard rather than one block.  Compute therefore waits on the full
    # shard window (the reader keeps the CB exactly full at every block boundary),
    # which is what preserves get_write_ptr() == get_read_ptr() for the in-place
    # rewrite when the L1 solve forces block_rows < shard_rows.
    in_wait_tiles = shard_tiles if (sharded and not is_row_major) else (B * S)
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
        RM_IN_DEPTH * S,
        1 if mask_enabled else 0,
        1 if sharded else 0,
        in_wait_tiles,
        in_shard_page,
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

    compute_ct = [
        S,
        B,
        S_COUNT,
        1 if has_gamma else 0,
        1 if is_row_major else 0,
        1 if mask_enabled else 0,
        in_wait_tiles,
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
