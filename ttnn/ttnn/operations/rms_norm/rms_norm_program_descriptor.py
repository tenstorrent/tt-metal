# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: blocking plan, CBs, kernels, runtime args (op_design.md).

Single source of truth for every block knob: `_select_blocking(...) -> BlockingPlan`.  Every CB page count,
loop trip count, core range and kernel argument below is derived from the plan's fields.

Regimes (op_design.md §Regimes):
  R1 row_local            (w_group_size == 1): each core owns whole rows; the row's W slice is resident.
  R2 w_split_root_combine (w_group_size >= 2): W split across a consecutive row-major group of cores; per block
                          every member sends its per-row partial sum of squares to the group root, the root
                          finalizes rsqrt(S/W + eps) and multicasts it back to the group.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
SCALER_TILE_BYTES = 2048  # bf16 scaler tile (reduce LLK scaler format)

# Selector constants (op_design.md §Selection function; perf lamps L2 / L3).
CHUNK_FLOOR_TILES = 8
MIN_W_TILES_PER_CORE = 8
X_DEPTH_BLOCKS = 1  # cb_x_tiles depth in blocks (perf lamp L1)

# CB indices (semantic names; op_design.md §Circular Buffers).
CB_X_RM_STICKS = 0
CB_X_TILES = 1
CB_GAMMA_RM_STICK = 2
CB_GAMMA_TILES = 3
CB_REDUCE_SCALER = 4
CB_SUMSQ = 5
CB_ROWSUM_PARTIAL = 6
CB_ROWSUM_GATHER = 7
CB_INV_RMS_ROOT = 8
CB_INV_RMS = 9
CB_X_GAMMA = 10
CB_OUT_TILES = 11
CB_OUT_RM_STICKS = 12

# Data-movement placement: reader on RISCV_1 / NOC_0, writer on RISCV_0 / NOC_1 (the ReaderConfig / WriterConfig
# defaults, spelled out because the inverse-RMS multicast is issued by the writer and must use the writer's NoC).
READER_NOC = ttnn.NOC.NOC_0
WRITER_NOC = ttnn.NOC.NOC_1

SEM_GATHER_ARRIVALS = 0
SEM_INV_MCAST_FIRST = 1  # Mcast attach allocates its semaphores from this id on


# ---------------------------------------------------------------------------------------------------------------------
# Geometry + blocking plan
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    tensor_w_elems: int
    tensor_w_tiles: int
    last_tile_valid_cols: int
    tensor_rows_tiles: int
    tensor_rows_sticks: int  # RM only (0 for TILE)
    is_rm: bool
    in_tile_bytes: int
    out_tile_bytes: int
    in_elem_bytes: int
    out_elem_bytes: int
    has_gamma: bool
    gamma_is_rm: bool
    gamma_tile_bytes: int
    gamma_elem_bytes: int
    gamma_is_block_float: bool  # bfloat8_b TILE gamma: the page is shared exponents + mantissas, read whole
    # DEST-produced intermediates (cb_sumsq, cb_rowsum_*, cb_inv_rms*, cb_x_gamma) follow DEST width: Float32 pages
    # under fp32_dest_acc_en=True, Float16_b pages under False (a 16-bit DEST value has no extra bits to keep).
    interm_dtype: object
    interm_tile_bytes: int
    # Sharded placement (Refinement 2). `sharded`: x / out live in L1 shards (same shard spec).
    # `zero_copy`: the shards back the CBs the kernels read x from / write out to — TILE: cb_x_tiles /
    # cb_out_tiles; RM: cb_x_rm_sticks (tilize source) / cb_out_rm_sticks (untilize destination), possible when
    # the shard's sticks are whole tile-rows of whole tiles (_rm_shard_zero_copy_ok). Otherwise (RM only) the
    # shard's sticks are restrided by local L1 copies into / out of those staging CBs.
    sharded: bool = False
    zero_copy: bool = False


@dataclass(frozen=True)
class BlockingPlan:
    w_group_size: int  # gw: cores per W-group (1 => R1)
    core_w_tiles: int  # n: W slice per core (tiles)
    num_row_groups: int  # gr
    rows_hi: int  # ceil(R / gr)
    block_rows: int  # b: tile-rows per block
    chunk_w_tiles: int  # c: streaming quantum along W (tiles)
    chunks_per_row: int  # k
    xg_lookahead_chunks: int  # L
    x_depth_blocks: int = X_DEPTH_BLOCKS
    # Gamma residency: False = the core's whole gamma slice is resident in cb_gamma_tiles (read once per kernel);
    # True = gamma streams one chunk at a time, double-buffered (re-read once per block). Sharded R1 only: the
    # shard pins the W slice, so a slice too wide for resident gamma has no wider W-group to fall back to.
    gamma_streamed: bool = False

    @property
    def row_stride_tiles(self):  # s = k * c
        return self.chunks_per_row * self.chunk_w_tiles

    @property
    def is_w_split(self):
        return self.w_group_size >= 2

    @property
    def used_cores(self):
        return self.num_row_groups * self.w_group_size


def _round_up(v, m):
    return ((v + m - 1) // m) * m


# CB page-format roles, resolved to a concrete dtype in _build_cbs.
_FMT_IN, _FMT_OUT, _FMT_GAMMA, _FMT_SCALER, _FMT_INTERM = "in", "out", "gamma", "scaler", "interm"
# CB backing: None = allocated in the CB region (counted by the L1 solve); _BACK_IN / _BACK_OUT = zero-copy on the
# input / output shard (ttnn.cb_descriptor_from_sharded_tensor) — already resident, so not counted.
_BACK_IN, _BACK_OUT = "input_shard", "output_shard"


def _cb_specs(geo: Geometry, b, n, c, k, L, gw, x_depth, gamma_streamed=False):
    """THE single definition of every CB: [(index, num_pages, page_bytes, format_role, backing)].

    Both the L1 solve (_footprint) and the allocation (_build_cbs) read this list, so a block / chunk / depth knob
    changes the budget and the real CB sizes in exactly one place (l1_ledger.md §Ledger mirrors it).
    """
    s = k * c
    out_depth = 1 if geo.is_rm else 2  # RM: normalize -> untilize are sequential on compute (no overlap to buy)
    stream_depth = 2  # reader / writer run one chunk ahead of compute
    tiles_backed = geo.zero_copy and not geo.is_rm
    sticks_backed = geo.zero_copy and geo.is_rm
    specs = []
    if geo.is_rm:
        specs.append(
            (CB_X_RM_STICKS, stream_depth * c, geo.in_tile_bytes, _FMT_IN, _BACK_IN if sticks_backed else None)
        )
    specs.append((CB_X_TILES, x_depth * b * s, geo.in_tile_bytes, _FMT_IN, _BACK_IN if tiles_backed else None))
    gamma_window_tiles = c if gamma_streamed else n  # tiles landed per gamma load
    if geo.has_gamma and geo.gamma_is_rm:
        # + one alignment unit: a sharded RM slice may start mid-alignment-unit of the gamma stick (read aligned down)
        align = ttnn.get_dram_alignment()
        stick_bytes = _round_up(gamma_window_tiles * TILE * geo.gamma_elem_bytes, align) + align
        specs.append((CB_GAMMA_RM_STICK, 1, stick_bytes, _FMT_GAMMA, None))
    if geo.has_gamma:
        gamma_pages = stream_depth * c if gamma_streamed else s
        specs.append((CB_GAMMA_TILES, gamma_pages, geo.gamma_tile_bytes, _FMT_GAMMA, None))
    specs.append((CB_REDUCE_SCALER, 2, SCALER_TILE_BYTES, _FMT_SCALER, None))
    specs.append((CB_SUMSQ, 2 * b, geo.interm_tile_bytes, _FMT_INTERM, None))
    if gw >= 2:
        specs.append((CB_ROWSUM_PARTIAL, b, geo.interm_tile_bytes, _FMT_INTERM, None))
        specs.append((CB_ROWSUM_GATHER, gw * b, geo.interm_tile_bytes, _FMT_INTERM, None))
        specs.append((CB_INV_RMS_ROOT, b, geo.interm_tile_bytes, _FMT_INTERM, None))
    specs.append((CB_INV_RMS, b, geo.interm_tile_bytes, _FMT_INTERM, None))
    if geo.has_gamma:
        specs.append((CB_X_GAMMA, L * b * c, geo.interm_tile_bytes, _FMT_INTERM, None))
    specs.append((CB_OUT_TILES, out_depth * b * c, geo.out_tile_bytes, _FMT_OUT, _BACK_OUT if tiles_backed else None))
    if geo.is_rm:
        specs.append(
            (
                CB_OUT_RM_STICKS,
                stream_depth * b * c,
                geo.out_tile_bytes,
                _FMT_OUT,
                _BACK_OUT if sticks_backed else None,
            )
        )
    return specs


def _footprint(geo: Geometry, b, n, c, k, L, gw, x_depth=X_DEPTH_BLOCKS, gamma_streamed=False):
    """Per-core CB bytes — the closed form of l1_ledger.md §Footprint, summed from _cb_specs."""
    return sum(
        pages * page_bytes
        for _, pages, page_bytes, _, backing in _cb_specs(geo, b, n, c, k, L, gw, x_depth, gamma_streamed)
        if backing is None
    )


def _fit_block(geo: Geometry, gw, n, gr, rows_hi, l1_budget_bytes):
    """Coarsest (chunk, lookahead, block_rows) for a fixed W-group / W slice — the one fit both selectors share.

    Zero-copy shards (geo.zero_copy) add wrap-safety constraints: x / out are read and written in place in the
    resident shard, whose row stride is exactly `n` tiles and whose height is `rows_hi` tile-rows, so the chunk must
    divide `n` (k·c == n; RM: k == 1, a tilize / untilize row is a whole shard stick), and a short last block runs
    at its actual row count (kernel CT `ragged_last_block`) so no nominal tail runs past the shard. Returns None
    if even the finest chunk does not fit.
    """
    if geo.zero_copy and geo.is_rm:
        chunk_candidates = [(n, 1)]
    elif geo.zero_copy:
        chunk_candidates = [(n // k, k) for k in range(1, n + 1) if n % k == 0]  # every divisor, largest first
    else:
        # largest balanced chunk first, down to the chunk floor (k <= n: beyond it the chunk stops shrinking)
        chunk_candidates = [(-(-n // k), k) for k in range(1, n + 1) if -(-n // k) >= min(n, CHUNK_FLOOR_TILES)]
    # Resident gamma first at every chunk; streamed gamma (sharded R1 only) only if no resident fit exists.
    gamma_modes = [False, True] if (geo.sharded and geo.has_gamma and gw == 1) else [False]
    for gamma_streamed in gamma_modes:
        for chunk, k in chunk_candidates:
            for L in [k, 1] if gw >= 2 else [1]:  # R2: whole-block x*gamma lookahead first (stall fill S1)
                fp = lambda b: _footprint(geo, b, n, chunk, k, L, gw, gamma_streamed=gamma_streamed)  # noqa: E731
                if fp(1) > l1_budget_bytes:
                    continue
                block_rows = 1  # k > 1 => 1 (M5: a chunk of a multi-row block is not FIFO-contiguous)
                if k == 1:
                    for b in range(rows_hi, 0, -1):
                        if fp(b) <= l1_budget_bytes:
                            block_rows = b
                            break
                return BlockingPlan(
                    w_group_size=gw,
                    core_w_tiles=n,
                    num_row_groups=gr,
                    rows_hi=rows_hi,
                    block_rows=block_rows,
                    chunk_w_tiles=chunk,
                    chunks_per_row=k,
                    xg_lookahead_chunks=L,
                    gamma_streamed=gamma_streamed,
                )
    return None


def _select_blocking(geo: Geometry, num_cores_available: int, l1_budget_bytes: int) -> BlockingPlan:
    """Interleaved placement: the op chooses the W-group (R1 / R2) and the row split."""
    C, R, Wt = num_cores_available, geo.tensor_rows_tiles, geo.tensor_w_tiles
    if R < C and C // R >= 2 and Wt >= 2 * MIN_W_TILES_PER_CORE:
        gw_fill = min(C // R, Wt // MIN_W_TILES_PER_CORE)
    else:
        gw_fill = 1
    seen = set()
    for gw_try in range(max(gw_fill, 1), C + 1):  # smallest W-group first: fewest combines
        n = -(-Wt // gw_try)
        gw = -(-Wt // n)  # M2: no empty slices
        if gw in seen:
            continue
        seen.add(gw)
        gr = min(R, C // gw)
        plan = _fit_block(geo, gw, n, gr, -(-R // gr), l1_budget_bytes)
        if plan is not None:
            return plan
    raise RuntimeError("rms_norm: W too wide for the grid")  # M3


def _rm_shard_zero_copy_ok(geo: Geometry, sp: "ShardPlacement"):
    """An RM shard can back the tilize source / untilize destination in place iff its sticks are whole tile-rows
    (shard height % 32 == 0) of whole tiles (shard width % 32 == 0, and W % 32 == 0 so no tile carries the shard's
    unspecified padding columns into the masked reduce)."""
    shard_h, shard_w = sp.shard_shape
    return geo.is_rm and shard_h % TILE == 0 and shard_w % TILE == 0 and geo.tensor_w_elems % TILE == 0


def _select_sharded_blocking(geo: Geometry, sp: "ShardPlacement", l1_budget_bytes: int):
    """Sharded placement: the shard spec pins the W-group, the row groups and the W slice (op_design.md
    §Placement axis) — HEIGHT -> R1 (gw = 1), WIDTH / BLOCK -> R2 (gw = shard columns). The shard is the block:
    `_fit_block` picks the coarsest block that fits, down from the whole resident shard. RM prefers the zero-copy
    stick CBs; it falls back to the local restride only for a geometry that cannot feed tilize in place.
    Returns (geometry, plan) — the geometry records which of the two RM realizations was chosen."""
    candidates = [replace(geo, zero_copy=True), geo] if _rm_shard_zero_copy_ok(geo, sp) else [geo]
    for g in candidates:
        plan = _fit_block(g, sp.num_shard_cols, sp.core_w_tiles, sp.num_shard_rows, sp.rows_hi, l1_budget_bytes)
        if plan is not None:
            return g, plan
    raise RuntimeError(
        f"rms_norm: sharded geometry (shard {sp.shard_shape}, {sp.num_shard_rows}x{sp.num_shard_cols} shards) "
        f"leaves {l1_budget_bytes} B of L1 for circular buffers, too little for one {sp.core_w_tiles}-tile slice"
    )


@dataclass(frozen=True)
class ShardPlacement:
    """The shard spec of a sharded input, resolved into the regime geometry (op_design.md §Placement axis).

    Shard (sr, sc) — shard-row sr (row group), shard-column sc (W-group member) — lives on cores[sr * cols + sc].
    `core_order` is the order in which every W-group is a consecutive run of `cores` (the Mcast grouping order).
    """

    memory_layout: object
    shard_shape: tuple  # (height, width) in elements (sticks x elements for RM)
    num_shard_rows: int
    num_shard_cols: int
    cores: tuple
    core_order: object
    core_w_tiles: int  # n: nominal W slice per core (tiles)
    rows_hi: int  # nominal tile-rows per core (TILE: the shard height; RM: ceil(sticks / 32))


def _shard_placement(input_tensor, geo: Geometry) -> ShardPlacement:
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    shard_h, shard_w = int(ss.shape[0]), int(ss.shape[1])
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    phys_h = geo.tensor_rows_sticks if geo.is_rm else geo.tensor_rows_tiles * TILE
    phys_w = int(input_tensor.padded_shape[-1])  # TILE: Wt*32; RM: the shard-padded stick width
    nsr, nsc = -(-phys_h // shard_h), -(-phys_w // shard_w)
    ml = mc.memory_layout
    if ml == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        start = ss.grid.bounding_box().start
        cores = [
            ttnn.CoreCoord(start.x + sc, start.y + sr) if row_major else ttnn.CoreCoord(start.x + sr, start.y + sc)
            for sr in range(nsr)
            for sc in range(nsc)
        ]
    else:
        cores = list(ttnn.corerange_to_cores(ss.grid, row_wise=row_major))[: nsr * nsc]
    if len(cores) != nsr * nsc or not all(ss.grid.contains(c) for c in cores):
        raise ValueError(f"rms_norm: shard grid {ss.grid} cannot hold {nsr}x{nsc} shards of {ss.shape}")
    order_key = (lambda c: (c.y, c.x)) if row_major else (lambda c: (c.x, c.y))
    if [order_key(c) for c in cores] != sorted(order_key(c) for c in cores):
        raise ValueError(f"rms_norm: shard grid {ss.grid} does not map shards to cores in {ss.orientation} order")
    if (nsc - 1) * shard_w >= geo.tensor_w_elems:
        raise ValueError(f"rms_norm: shard width {shard_w} leaves the last shard column without valid elements")
    if geo.is_rm:
        n = -(-shard_w // TILE)
        rows_hi = -(-min(shard_h, phys_h) // TILE)
    else:
        n = shard_w // TILE
        rows_hi = shard_h // TILE
    return ShardPlacement(
        memory_layout=ml,
        shard_shape=(shard_h, shard_w),
        num_shard_rows=nsr,
        num_shard_cols=nsc,
        cores=tuple(cores),
        core_order=ttnn.McastCoreOrder.RowMajor if row_major else ttnn.McastCoreOrder.ColumnMajor,
        core_w_tiles=n,
        rows_hi=rows_hi,
    )


def _cores_to_corerangeset(cores):
    """CoreRangeSet over `cores`, one range per run of x-consecutive cores in a row."""
    by_row = {}
    for c in cores:
        by_row.setdefault(c.y, []).append(c.x)
    ranges = set()
    for y, xs in by_row.items():
        xs = sorted(xs)
        run_start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            ranges.add(ttnn.CoreRange(ttnn.CoreCoord(run_start, y), ttnn.CoreCoord(prev, y)))
            if x is not None:
                run_start = prev = x
    return ttnn.CoreRangeSet(ranges)


_BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def _elem_bytes(tensor):
    """Bytes per element, consumed only by the ROW_MAJOR stick paths and the TILE-gamma face-row reads.

    A block-float tensor (TILE-only, shared exponents) has no per-element size and never takes those paths, so 0.
    """
    return 0 if tensor.dtype in _BLOCK_FLOAT_DTYPES else tensor.element_size()


def _geometry(input_tensor, output_tensor, gamma, fp32_dest_acc_en) -> Geometry:
    shape = list(input_tensor.shape)
    W = shape[-1]
    H = shape[-2]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    w_tiles = -(-W // TILE)
    if is_rm:
        rows_sticks = lead * H
        rows_tiles = -(-rows_sticks // TILE)
    else:
        rows_sticks = 0
        rows_tiles = lead * (-(-H // TILE))
    has_gamma = gamma is not None
    interm_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    sharded = input_tensor.memory_config().is_sharded()
    return Geometry(
        tensor_w_elems=W,
        tensor_w_tiles=w_tiles,
        last_tile_valid_cols=W - TILE * (w_tiles - 1),
        tensor_rows_tiles=rows_tiles,
        tensor_rows_sticks=rows_sticks,
        is_rm=is_rm,
        in_tile_bytes=ttnn.tile_size(input_tensor.dtype),
        out_tile_bytes=ttnn.tile_size(output_tensor.dtype),
        in_elem_bytes=_elem_bytes(input_tensor),
        out_elem_bytes=_elem_bytes(output_tensor),
        has_gamma=has_gamma,
        gamma_is_rm=has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT,
        gamma_tile_bytes=ttnn.tile_size(gamma.dtype) if has_gamma else 0,
        gamma_elem_bytes=_elem_bytes(gamma) if has_gamma else 0,
        gamma_is_block_float=has_gamma and gamma.dtype in _BLOCK_FLOAT_DTYPES,
        interm_dtype=interm_dtype,
        interm_tile_bytes=ttnn.tile_size(interm_dtype),
        sharded=sharded,
        zero_copy=sharded and not is_rm,
    )


def _l1_budget_bytes(device):
    """Worker-L1 bytes available to CBs: from the allocator base up to the lowest allocated L1 buffer.

    CBs grow up from the L1 allocator base (``GetAllocatorBaseAddress``) while L1 buffers (including any
    L1-interleaved input / output / gamma) are allocated top-down, so the budget is the free run at the bottom of
    the bank as reported by the allocator's memory view. No safety fraction.
    """
    view = ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.L1)
    budget = view.total_bytes_per_bank
    for block in view.block_table:
        if block["allocated"] == "yes":
            budget = min(budget, int(block["address"]))
    return budget


# ---------------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------------


def _cb(index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _build_cbs(geo: Geometry, plan: BlockingPlan, cores, input_tensor, output_tensor, gamma_dtype):
    """Every CB declared once over all used cores at one nominal size (M1), from the same _cb_specs as the solve.

    Shard-backed CBs are globally allocated on the shard (zero-copy), so they take no CB-region L1 and do not move
    the CB-region addresses of the remote-written cb_rowsum_gather / cb_inv_rms (identical on every core, M1).
    """
    fmt = {
        _FMT_IN: input_tensor.dtype,
        _FMT_OUT: output_tensor.dtype,
        _FMT_GAMMA: gamma_dtype,
        _FMT_SCALER: ttnn.bfloat16,
        _FMT_INTERM: geo.interm_dtype,
    }
    backing_tensor = {_BACK_IN: input_tensor, _BACK_OUT: output_tensor}
    specs = _cb_specs(
        geo,
        plan.block_rows,
        plan.core_w_tiles,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        plan.xg_lookahead_chunks,
        plan.w_group_size,
        plan.x_depth_blocks,
        plan.gamma_streamed,
    )
    cbs = []
    for index, pages, page_bytes, role, backing in specs:
        if backing is None:
            cbs.append(_cb(index, cores, pages, page_bytes, fmt[role]))
        else:
            # Zero-copy on the shard: the buffer, total size and cores come from the shard; the page is the CB's
            # own unit (a tile, or 32 sticks x one tile-column for the RM stick CBs) — not the buffer's page.
            desc = ttnn.cb_descriptor_from_sharded_tensor(index, backing_tensor[backing], core_ranges=cores)
            desc.format_descriptors = [
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt[role], page_size=page_bytes)
            ]
            cbs.append(desc)
    return cbs


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@dataclass(frozen=True)
class CoreWork:
    """One core's share of the plan — every per-core runtime argument is derived from these fields.

    Rows are tile-rows of the core's row space (TILE interleaved / RM: global tile-row index from `tile_row_start`;
    TILE shard: the shard's own rows). RM sticks of that space are valid below `row_sticks_end`. The W slice is
    `slice_valid_elems` valid elements starting at `x_col0_bytes` / `out_col0_bytes` of each source / destination
    stick (RM), at global tile-column `slice_start_tile` (TILE) and global element `gamma_col0_elem` (gamma).
    """

    core: object
    member_idx: int
    root: object
    tile_row_start: int
    rows_tiles: int
    row_sticks_end: int
    slice_start_tile: int
    slice_w_tiles: int
    slice_valid_elems: int
    x_col0_bytes: int
    out_col0_bytes: int
    gamma_col0_elem: int

    @property
    def last_tile_valid_cols(self):
        return self.slice_valid_elems - TILE * (self.slice_w_tiles - 1)


def _interleaved_work(geo: Geometry, plan: BlockingPlan, grid):
    """Interleaved: consecutive row-major runs of the grid form the W-groups; rows split two-class over groups."""
    gw, gr = plan.w_group_size, plan.num_row_groups
    core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(plan.used_cores)]
    q, rem = divmod(geo.tensor_rows_tiles, gr)  # split_work_to_cores semantics (two classes)
    work = []
    for i, core in enumerate(core_list):
        gi, g = divmod(i, gw)
        rows = q + 1 if gi < rem else q
        slice_start = g * plan.core_w_tiles
        slice_w = min(plan.core_w_tiles, geo.tensor_w_tiles - slice_start)
        valid_elems = min(slice_w * TILE, geo.tensor_w_elems - slice_start * TILE)
        work.append(
            CoreWork(
                core=core,
                member_idx=g,
                root=core_list[gi * gw],
                tile_row_start=gi * q + min(gi, rem),
                rows_tiles=rows,
                row_sticks_end=geo.tensor_rows_sticks,
                slice_start_tile=slice_start,
                slice_w_tiles=slice_w,
                slice_valid_elems=valid_elems,
                x_col0_bytes=slice_start * TILE * geo.in_elem_bytes,
                out_col0_bytes=slice_start * TILE * geo.out_elem_bytes,
                gamma_col0_elem=slice_start * TILE,
            )
        )
    return ttnn.num_cores_to_corerangeset(plan.used_cores, grid, True), work


def _sharded_work(geo: Geometry, plan: BlockingPlan, sp: ShardPlacement):
    """Sharded: shard (sr, sc) is row group sr, W-group member sc; the core's rows / slice are its shard's."""
    shard_h, shard_w = sp.shard_shape
    work = []
    for i, core in enumerate(sp.cores):
        sr, sc = divmod(i, sp.num_shard_cols)
        if geo.is_rm:
            valid_sticks = min(shard_h, geo.tensor_rows_sticks - sr * shard_h)
            valid_elems = min(shard_w, geo.tensor_w_elems - sc * shard_w)
            # zero-copy: walk the whole nominal shard, like TILE (all W-group members run the same blocks)
            rows = sp.rows_hi if geo.zero_copy else -(-valid_sticks // TILE)
            slice_w = -(-valid_elems // TILE)
            gamma_col0 = sc * shard_w
        else:
            # Zero-copy: every core walks its whole nominal shard (padding rows are row-local, never written back
            # as valid data), so all members of a W-group run the same number of blocks.
            valid_sticks = 0
            rows = sp.rows_hi
            slice_w = min(sp.core_w_tiles, geo.tensor_w_tiles - sc * sp.core_w_tiles)
            valid_elems = min(slice_w * TILE, geo.tensor_w_elems - sc * sp.core_w_tiles * TILE)
            gamma_col0 = sc * sp.core_w_tiles * TILE
        work.append(
            CoreWork(
                core=core,
                member_idx=sc,
                root=sp.cores[sr * sp.num_shard_cols],
                tile_row_start=0,
                rows_tiles=rows,
                row_sticks_end=valid_sticks,
                slice_start_tile=gamma_col0 // TILE,
                slice_w_tiles=slice_w,
                slice_valid_elems=valid_elems,
                x_col0_bytes=0,
                out_col0_bytes=0,
                gamma_col0_elem=gamma_col0,
            )
        )
    return _cores_to_corerangeset(sp.cores), work


def create_program_descriptor(input_tensor, output_tensor, *, gamma, epsilon, compute_kernel_config):
    device = input_tensor.device()
    geo = _geometry(input_tensor, output_tensor, gamma, bool(compute_kernel_config.fp32_dest_acc_en))
    grid = device.compute_with_storage_grid_size()
    l1_budget = _l1_budget_bytes(device)
    if geo.sharded:
        sp = _shard_placement(input_tensor, geo)
        geo, plan = _select_sharded_blocking(geo, sp, l1_budget)
        cores, work = _sharded_work(geo, plan, sp)
        core_order = sp.core_order
    else:
        plan = _select_blocking(geo, grid.x * grid.y, l1_budget)
        cores, work = _interleaved_work(geo, plan, grid)
        core_order = ttnn.McastCoreOrder.RowMajor
    gw = plan.w_group_size

    cbs = _build_cbs(
        geo,
        plan,
        cores,
        input_tensor,
        output_tensor,
        gamma.dtype if gamma is not None else ttnn.bfloat16,
    )

    # How the reader sources x and the writer sinks out (kernel CT arg): 0 = interleaved (TensorAccessor),
    # 1 = local shard restride (RM), 2 = zero-copy (the shard backs the CB; no data movement).
    x_out_mode = 0 if not geo.sharded else (2 if geo.zero_copy else 1)
    # RM restride: local stick stride of the input / output shard = the buffer's aligned page (one padded stick).
    shard_stick_bytes_in = shard_stick_bytes_out = 0
    if geo.sharded and geo.is_rm:
        shard_stick_bytes_in = ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor).format_descriptors[0].page_size
        shard_stick_bytes_out = ttnn.cb_descriptor_from_sharded_tensor(0, output_tensor).format_descriptors[0].page_size

    # ---------------- reader ----------------
    reader_ct = [
        CB_X_RM_STICKS,
        CB_X_TILES,
        CB_REDUCE_SCALER,
        int(geo.is_rm),
        plan.block_rows,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        geo.in_elem_bytes,
        geo.tensor_w_tiles,
        x_out_mode,
        shard_stick_bytes_in,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- writer ----------------
    # TILE gamma: row 0 of faces 0 / 1 by byte offset (16 elements per face row) — or, for a bfloat8_b page whose
    # shared exponents precede the mantissas, the whole page (the Row broadcast still consumes only row 0).
    face_row_path = geo.has_gamma and not geo.gamma_is_rm and not geo.gamma_is_block_float
    gamma_face_row_read_bytes = _round_up(16 * geo.gamma_elem_bytes, ttnn.get_dram_alignment()) if face_row_path else 0
    writer_ct = [
        CB_GAMMA_RM_STICK,
        CB_GAMMA_TILES,
        CB_ROWSUM_PARTIAL,
        CB_ROWSUM_GATHER,
        CB_INV_RMS_ROOT,
        CB_INV_RMS,
        CB_OUT_TILES,
        CB_OUT_RM_STICKS,
        int(geo.is_rm),
        int(geo.has_gamma),
        int(geo.gamma_is_rm),
        int(plan.is_w_split),
        plan.block_rows,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        gw,
        geo.out_elem_bytes,
        geo.gamma_elem_bytes,
        geo.tensor_w_tiles,
        geo.tensor_w_elems,
        x_out_mode,
        SEM_GATHER_ARRIVALS,
        gamma_face_row_read_bytes,
        int(geo.gamma_is_block_float),
        shard_stick_bytes_out,
        ttnn.get_dram_alignment(),
        int(plan.gamma_streamed),
        int(geo.zero_copy),  # ragged_last_block
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # ---------------- compute ----------------
    compute_ct = [
        CB_X_RM_STICKS,
        CB_X_TILES,
        CB_GAMMA_TILES,
        CB_REDUCE_SCALER,
        CB_SUMSQ,
        CB_ROWSUM_PARTIAL,
        CB_ROWSUM_GATHER,
        CB_INV_RMS_ROOT,
        CB_INV_RMS,
        CB_X_GAMMA,
        CB_OUT_TILES,
        CB_OUT_RM_STICKS,
        int(geo.is_rm),
        int(geo.has_gamma),
        int(plan.is_w_split),
        plan.block_rows,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        plan.xg_lookahead_chunks,
        gw,
        int(plan.gamma_streamed),
        int(geo.zero_copy),  # ragged_last_block
    ]

    inv_w_bits = _f32_bits(1.0 / geo.tensor_w_elems)
    eps_bits = _f32_bits(epsilon)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for w in work:
        x, y = w.core.x, w.core.y
        num_blocks = -(-w.rows_tiles // plan.block_rows)
        root_virtual = device.worker_core_from_logical_core(w.root)
        reader_rt[x][y] = [
            input_tensor.buffer_address(),
            w.tile_row_start,
            w.rows_tiles,
            w.slice_start_tile,
            w.slice_w_tiles,
            w.last_tile_valid_cols,
            w.row_sticks_end,
            w.slice_valid_elems,
            w.x_col0_bytes,
        ]
        writer_rt[x][y] = [
            output_tensor.buffer_address(),
            gamma.buffer_address() if gamma is not None else 0,
            w.tile_row_start,
            w.rows_tiles,
            w.slice_start_tile,
            w.slice_w_tiles,
            w.member_idx,
            root_virtual.x,
            root_virtual.y,
            w.row_sticks_end,
            w.slice_valid_elems,
            w.out_col0_bytes,
            w.gamma_col0_elem,
        ]
        compute_rt[x][y] = [num_blocks, w.slice_w_tiles, int(w.member_idx == 0), inv_w_bits, eps_bits, w.rows_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=WRITER_NOC),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_GATHER_ARRIVALS, core_ranges=cores, initial_value=0)]
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    if plan.is_w_split:
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=WRITER_NOC, handshake=False),
            cores,
            gw,
            ttnn.McastFixedSenderConfig(sender_index=0),
            core_order,
        )
        mcast.attach(descriptor, "inv_mcast", [writer], SEM_INV_MCAST_FIRST)
    else:
        ttnn.attach_absent_mcast(writer, "inv_mcast")
    descriptor.kernels = [reader, writer, compute]
    return descriptor
