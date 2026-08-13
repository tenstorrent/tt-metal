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
DM_CHUNK_TILES = 8  # tiles per NoC barrier, reader and writer alike

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
CB_SLICE_STAT = 3
CB_GATHERED_PARTIALS = 4
CB_RMS_BCAST = 5
CB_RMS_RECIP = 6
CB_SCALER = 7
CB_W_MASK = 8
CB_OUTPUT_TILES = 9
CB_RM_STAGE_IN = 10
CB_RM_STAGE_OUT = 11

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
        total += b * bytes_["stat_tile"]  # cb_slice_stat
        total += s * b * bytes_["stat_tile"]  # cb_gathered_partials
        total += b * bytes_["stat_tile"]  # cb_rms_bcast
    total += b * bytes_["stat_tile"]  # cb_rms_recip
    total += 2 * bytes_["bf16_tile"]  # cb_scaler + cb_w_mask
    if is_row_major:
        total += RM_IN_DEPTH * s_ * bytes_["in_tile"]  # cb_rm_stage_in
        total += RM_OUT_DEPTH * s_ * bytes_["out_tile"]  # cb_rm_stage_out
    else:
        total += OUT_CB_DEPTH * s_ * bytes_["out_tile"]  # cb_output_tiles
    return total


# ---------------------------------------------------------------------------
# Work distribution / regime selection (op_design.md §Work Distribution)
# ---------------------------------------------------------------------------


def _slice_candidates(gx, gy):
    """Slice counts that tile the grid into exact rectangles.

    s <= gx  -> rect is (s x 1) and s must divide gx.
    s >  gx  -> rect is (gx x rect_h), so s must be gx * rect_h with rect_h <= gy.
    """
    out = set()
    for s in range(1, gx + 1):
        if gx % s == 0:
            out.add(s)
    for rect_h in range(2, gy + 1):
        out.add(gx * rect_h)
    return sorted(out)


def _plan(device, input_tensor, *, has_gamma, bytes_):
    """Pick (num_row_groups, num_hidden_slices, slice_hidden_tiles, block_rows)."""
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    row_tiles, hidden_tiles, _w, _sticks = _tile_geometry(input_tensor)
    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    budget = _l1_working_budget(device)

    # Hidden-granularity floor: never slice so thin that the combine dominates a
    # 1-tile payload (feature_spec.py:343-346 measured geometries land at 4-8).
    slice_cap = min(hidden_tiles, max(1, _div_up(hidden_tiles, HIDDEN_TILES_PER_CORE_FLOOR)), gx * gy)

    best = None
    for s in _slice_candidates(gx, gy):
        if s > slice_cap:
            continue
        slice_tiles = _div_up(hidden_tiles, s)
        # Tightness: every slice must own at least one tile (no idle rect core).
        if _div_up(hidden_tiles, slice_tiles) != s:
            continue
        rect_w = min(s, gx)
        if s % rect_w != 0 or gx % rect_w != 0:
            continue
        rect_h = s // rect_w
        if rect_h > gy:
            continue
        if _footprint_bytes(1, slice_tiles, s, is_row_major=is_row_major, has_gamma=has_gamma, bytes_=bytes_) > budget:
            continue
        groups_geom = (gx // rect_w) * (gy // rect_h)
        groups = min(groups_geom, max(1, row_tiles))
        cores = groups * s
        # Occupancy first; tie-break toward FEWER slices (fewer combines).
        score = (cores, -s)
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

    shape = list(input_tensor.shape)
    W = shape[-1]
    in_elem = input_tensor.element_size()
    out_elem = output_tensor.element_size()
    gamma_elem = gamma.element_size() if has_gamma else in_elem
    total_sticks = input_tensor.buffer_num_pages() if is_row_major else 0

    mask_valid_w = W % TILE_DIM
    mask_enabled = (mask_valid_w != 0) and not is_row_major

    # ---- core assignment: G rectangles of (rect_w x rect_h) cores ----
    rects_per_grid_row = gx // rect_w
    row_base, row_rem = divmod(Rt, G)

    groups = []  # list of dicts: origin, cores (row-major within the rect), rows
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
                "num_blocks": _div_up(rows, B) if rows > 0 else 0,
            }
        )
        row_cursor += rows

    active_groups = [g for g in groups if g["core_row_tiles"] > 0]
    # Each row-group IS a rectangle, so the union is one CoreRange per group.
    all_cores_crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(g["origin"][0], g["origin"][1]),
                ttnn.CoreCoord(g["origin"][0] + rect_w - 1, g["origin"][1] + rect_h - 1),
            )
            for g in active_groups
        ]
    )

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
            rect_crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + rect_w - 1, oy + rect_h - 1))]
            )
            mcast_by_group[idx] = ttnn.Mcast2D(device, rect_crs, ttnn.CoreCoord(ox, oy), cfg)
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
            core_ranges=all_cores_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        # capacity == live set exactly: the in-place rewrite of x needs
        # get_write_ptr == get_read_ptr, which only holds at a full CB.
        _cb(CB_INPUT_TILES, IN_CB_DEPTH * B * S, in_tile, input_tensor.dtype),
        _cb(CB_SQ_PARTIALS, B, stat_tile, ttnn.float32),
        _cb(CB_RMS_RECIP, B, stat_tile, ttnn.float32),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16),
        _cb(CB_W_MASK, 1, bf16_tile, ttnn.bfloat16),
    ]
    if has_gamma:
        cbs.append(_cb(CB_GAMMA_TILES, S, gamma_tile, gamma.dtype))
    if S_COUNT > 1:
        cbs.append(_cb(CB_SLICE_STAT, B, stat_tile, ttnn.float32))
        cbs.append(_cb(CB_GATHERED_PARTIALS, S_COUNT * B, stat_tile, ttnn.float32))
        cbs.append(_cb(CB_RMS_BCAST, B, stat_tile, ttnn.float32))
    if is_row_major:
        cbs.append(_cb(CB_RM_STAGE_IN, RM_IN_DEPTH * S, in_tile, input_tensor.dtype))
        cbs.append(_cb(CB_RM_STAGE_OUT, RM_OUT_DEPTH * S, out_tile, output_tensor.dtype))
    else:
        cbs.append(_cb(CB_OUTPUT_TILES, OUT_CB_DEPTH * S, out_tile, output_tensor.dtype))

    # ---- compile-time args ----
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
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [
        S,
        B,
        S_COUNT,
        1 if has_gamma else 0,
        1 if is_row_major else 0,
        1 if mask_enabled else 0,
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
            slice_base = slice_index * S
            valid_tiles = max(0, min(S, Wt - slice_base))
            valid_w = max(0, min(S * TILE_DIM, W - slice_base * TILE_DIM))
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
                total_sticks,
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
                total_sticks,
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
        core_ranges=all_cores_crs,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores_crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores_crs,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_cores_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_cores_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_GATHER_PROGRESS, core_ranges=all_cores_crs, initial_value=0),
    ]

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
