# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""genify_matmul program descriptor — regime R1 `interleaved_out_split` (op_design.md, Blocking Model).

Host side of the blocking scheme:
  * choose_split()    — the core-assignment knob: logical (grid_b, grid_m, grid_n) factorization.
                        Candidates within (1 + GRID_W_SLACK) of the best per-core output tiles W are
                        blocked (choose_blocking) and ranked by (aggregate read traffic incl. block
                        re-reads, W, cores).
  * choose_blocking() — the extent knobs (block_batches, block_m_tiles, block_n_tiles, block_k_tiles /
                        num_k_blocks / last_block_k_tiles), solved coarsest-first against the L1 budget
                        (steps S1..S4), plus the derived output subblock (sb_h, sb_w).
Every CB page count, kernel loop trip count and grid size below derives from those knobs; each knob is
defined once and handed to the kernels as a CT / RT arg.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# ---- CB indices (op_design.md, Circular Buffers) ----
CB_ACT_TILES = 0  # A K-blocks, reader -> compute
CB_WEIGHT_TILES = 1  # B K-blocks, reader -> compute
CB_BIAS_ROW = 2  # bias N-block row (bias / bias_relu), reader -> compute, one per block
CB_GATE_ROW = 3  # gate N-block row (addcmul), reader -> compute, one per block
CB_RESIDUAL_TILES = 4  # residual row groups (addcmul), reader -> compute
CB_OUTPUT_TILES = 16  # output row groups, compute -> writer
CB_PARTIALS = 24  # K-accumulator (packer L1 acc), compute -> compute; num_k_blocks > 1 or epilogue != none

# ---- Epilogue kinds (compile-time arg shared by reader and compute) ----
EPILOGUES = {"none": 0, "bias": 1, "bias_relu": 2, "swiglu": 3, "addcmul": 4}

# ---- Buffer-depth knobs ----
DEPTH_ACT = 2  # A K-blocks in flight
DEPTH_WEIGHT = 2  # B K-blocks in flight
DEPTH_OUT = 2  # output row groups in flight
DEPTH_RESIDUAL = 2  # residual row groups in flight (addcmul)
# Epilogue chain DEST lanes per acquire (eltwise_chain block_size; the chain clamps it to DEST).
EPILOGUE_BLOCK_TILES = 1

# ---- Blocking knobs ----
BLOCK_K_FLOOR = 8  # smallest K-block the solve accepts (clamped to tensor_k_tiles)
# Precision knob, 16-bit DEST only: the longest K chain accumulated inside DEST before the packer
# folds it into cb_partials (packer L1 accumulation). A bf16 DEST running sum swamps small addends,
# so the error grows with the in-DEST chain: K = 16384 as one 171-tile K-block measured rel-RMS 0.185
# (stock ttnn.matmul 0.027); capped at 8 tiles 0.019 (32 tiles: 0.046). fp32 DEST is uncapped.
DEST16_BLOCK_K_MAX = 8
Q_N_MAX = 4  # subblock quantum on core_n_tiles (a 4-tile subblock always exists)
# Core-assignment knob: a factorization whose per-core output tiles W are within (1 + GRID_W_SLACK)
# of the best W competes on aggregate DRAM read traffic (R summed over the used cores). R1 has no
# operand reuse across cores, so it is DRAM-bound on every measured shape (HiFi2 ~= HiFi4); ranking
# W strictly first picked e.g. a 1x108 N-only split for M=9472, K=5120, N=3456 (W 296 vs 324) that
# made every core read all of A (10.5 GB instead of 1.3 GB, 4.5x slower).
GRID_W_SLACK = 0.25
L1_RESERVE_BYTES = 0  # L1 held back for foreign tensors


def _cdiv(a, b):
    return -(-a // b)


def _round_up(a, b):
    return _cdiv(a, b) * b


def _pow2_floor(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _divisors(x):
    return [d for d in range(1, x + 1) if x % d == 0]


@dataclass(frozen=True)
class Geometry:
    tensor_batches: int
    tensor_m_tiles: int
    tensor_k_tiles: int
    tensor_n_tiles: int
    out_n_tiles: int
    weight_batch_stride: int
    a_tile_bytes: int
    b_tile_bytes: int
    o_tile_bytes: int
    p_tile_bytes: int
    p_dtype: object
    dest_limit: int
    block_k_cap: int  # longest in-DEST K chain (DEST16_BLOCK_K_MAX with 16-bit DEST, else all of K)
    epilogue: str = "none"
    row_tile_bytes: int = 0  # bias (bias / bias_relu) or gate (addcmul) tile bytes, else 0
    r_tile_bytes: int = 0  # residual tile bytes (addcmul), else 0

    @property
    def swiglu(self):
        return self.epilogue == "swiglu"

    @property
    def q_n_min(self):
        # swiglu: whole (gate, up) tile pairs per core and per block (N % 64 == 0 makes Nt even).
        return 2 if self.swiglu else 1

    def out_block_n(self, block_n_tiles):
        return block_n_tiles // 2 if self.swiglu else block_n_tiles


@dataclass(frozen=True)
class GridChoice:
    grid_b: int  # used logical grid along each axis
    grid_m: int
    grid_n: int
    core_batches: int
    core_m_tiles: int
    core_n_tiles: int
    q_n: int
    placement: str  # "rect" (x=n, y=m), "rect_t" (x=m, y=n) or "flat" (row-wise list)


@dataclass(frozen=True)
class Blocking:
    block_batches: int
    block_m_tiles: int
    block_n_tiles: int
    block_k_tiles: int
    num_k_blocks: int
    last_block_k_tiles: int
    sb_h: int
    sb_w: int
    epilogue: str = "none"

    @property
    def use_partials(self):
        # The K-accumulator exists for a multi-K-block walk, and always under an epilogue (the
        # matmul packs its last K-block there for the epilogue to read: LastBlockTarget::Interm).
        return self.num_k_blocks > 1 or self.epilogue != "none"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _geometry(a, b, out, fp32_dest_acc_en, epilogue="none", row=None, residual=None):
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    lead = a_shape[:-2]
    img_m_tiles = _cdiv(a_shape[-2], TILE)
    tensor_k_tiles = _cdiv(a_shape[-1], TILE)
    tensor_n_tiles = _cdiv(b_shape[-1], TILE)
    out_n_tiles = _cdiv(list(out.shape)[-1], TILE)
    lead_prod = 1
    for d in lead:
        lead_prod *= d
    batched = any(d > 1 for d in b_shape[:-2])
    if batched:
        tensor_batches = lead_prod
        tensor_m_tiles = img_m_tiles
        weight_batch_stride = tensor_k_tiles * tensor_n_tiles
    else:
        # Single B: the leading dims fold into M (each image is tile-padded on its own).
        tensor_batches = 1
        tensor_m_tiles = lead_prod * img_m_tiles
        weight_batch_stride = 0
    p_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    return Geometry(
        tensor_batches=tensor_batches,
        tensor_m_tiles=tensor_m_tiles,
        tensor_k_tiles=tensor_k_tiles,
        tensor_n_tiles=tensor_n_tiles,
        out_n_tiles=out_n_tiles,
        weight_batch_stride=weight_batch_stride,
        a_tile_bytes=ttnn.tile_size(a.dtype),
        b_tile_bytes=ttnn.tile_size(b.dtype),
        o_tile_bytes=ttnn.tile_size(out.dtype),
        p_tile_bytes=ttnn.tile_size(p_dtype),
        p_dtype=p_dtype,
        dest_limit=4 if fp32_dest_acc_en else 8,
        block_k_cap=tensor_k_tiles if fp32_dest_acc_en else min(tensor_k_tiles, DEST16_BLOCK_K_MAX),
        epilogue=epilogue,
        row_tile_bytes=ttnn.tile_size(row.dtype) if row is not None else 0,
        r_tile_bytes=ttnn.tile_size(residual.dtype) if residual is not None else 0,
    )


# ---------------------------------------------------------------------------
# Work split (core-assignment knob)
# ---------------------------------------------------------------------------


def _placement_for(gb, gm, gn, gx, gy):
    if gb == 1 and gm <= gy and gn <= gx:
        return "rect"
    if gb == 1 and gm <= gx and gn <= gy:
        return "rect_t"
    if gb > 1 or gm == 1 or gn == 1:
        return "flat"
    return None


def grid_candidates(geom: Geometry, gx: int, gy: int):
    """Every legal (grid_b, grid_m, grid_n) factorization as a GridChoice (used grid, deduplicated)."""
    C = gx * gy
    tb, tm, tn = geom.tensor_batches, geom.tensor_m_tiles, geom.tensor_n_tiles
    seen = {}
    for gb in range(1, min(tb, C) + 1):
        for gm in range(1, min(tm, C // gb) + 1):
            for gn in range(1, min(tn, C // (gb * gm)) + 1):
                core_b = _cdiv(tb, gb)
                core_m = _cdiv(tm, gm)
                n_per = _cdiv(tn, gn)
                q_n = max(geom.q_n_min, min(Q_N_MAX, _pow2_floor(n_per)))
                core_n = _round_up(n_per, q_n)
                ub, um, un = _cdiv(tb, core_b), _cdiv(tm, core_m), _cdiv(tn, core_n)
                placement = _placement_for(ub, um, un, gx, gy)
                if placement is None:
                    continue
                grid = GridChoice(ub, um, un, core_b, core_m, core_n, q_n, placement)
                seen.setdefault((ub, um, un, core_b, core_m, core_n, q_n), grid)
    return list(seen.values())


def grid_work(grid: GridChoice):
    """W: per-core output tiles, the critical path (each costs tensor_k_tiles matmul steps)."""
    return grid.core_batches * grid.core_m_tiles * grid.core_n_tiles


def read_traffic(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """Aggregate A + B bytes read over the used cores, including the re-reads the block split adds
    (A once per N-block, B once per M-block of each core's region)."""
    nb_m = _cdiv(grid.core_m_tiles, blk.block_m_tiles)
    nb_n = _cdiv(grid.core_n_tiles, blk.block_n_tiles)
    per_core = (
        grid.core_batches
        * geom.tensor_k_tiles
        * (grid.core_m_tiles * geom.a_tile_bytes * nb_n + grid.core_n_tiles * geom.b_tile_bytes * nb_m)
    )
    return per_core * grid.grid_b * grid.grid_m * grid.grid_n


def choose_split(geom: Geometry, gx: int, gy: int, budget: int):
    """Core-assignment knob + extent knobs together. Among the factorizations whose W is within
    (1 + GRID_W_SLACK) of the best W, solve the blocking of each and take the lexicographic min of
    (aggregate read traffic incl. block re-reads, W, used cores)."""
    candidates = grid_candidates(geom, gx, gy)
    w_cap = min(grid_work(g) for g in candidates) * (1 + GRID_W_SLACK)
    best_key, best = None, None
    for grid in candidates:
        if grid_work(grid) > w_cap:
            continue
        try:
            blk = choose_blocking(geom, grid, budget)
        except RuntimeError:
            continue
        key = (read_traffic(geom, grid, blk), grid_work(grid), grid.grid_b * grid.grid_m * grid.grid_n)
        if best_key is None or key < best_key:
            best_key, best = key, (grid, blk)
    if best is None:
        raise RuntimeError(f"genify_matmul: even the minimum block does not fit the L1 CB budget of {budget} bytes")
    return best


def _logical_cores(grid: GridChoice, gx: int):
    """[(b, i, j, CoreCoord)] for every used logical region, plus the CoreRangeSet covering them."""
    out = []
    for b in range(grid.grid_b):
        for i in range(grid.grid_m):
            for j in range(grid.grid_n):
                if grid.placement == "rect":
                    x, y = j, i
                elif grid.placement == "rect_t":
                    x, y = i, j
                else:
                    p = (b * grid.grid_m + i) * grid.grid_n + j
                    x, y = p % gx, p // gx
                out.append((b, i, j, ttnn.CoreCoord(x, y)))
    if grid.placement == "rect":
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.grid_n - 1, grid.grid_m - 1))]
    elif grid.placement == "rect_t":
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.grid_m - 1, grid.grid_n - 1))]
    else:
        n = len(out)
        full_rows, rem = n // gx, n % gx
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return out, ttnn.CoreRangeSet(ranges)


# ---------------------------------------------------------------------------
# Blocking (extent knobs, L1 solve)
# ---------------------------------------------------------------------------


def choose_subblock(block_m, block_n, dest_limit):
    """Largest sb_h*sb_w <= dest_limit with sb_h | block_m, sb_w | block_n; ties go to larger sb_w."""
    best = (1, 1)
    for h in _divisors(block_m):
        for w in _divisors(block_n):
            if h * w <= dest_limit and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _footprint(geom, bm, bn, bk, partials):
    """F(block_m, block_n, block_k) of l1_ledger.md, epilogue terms included (E_bias, E_addcmul).
    An epilogue always needs cb_partials (the matmul's last K-block lands there)."""
    sb_h, _ = choose_subblock(bm, bn, geom.dest_limit)
    partials = partials or geom.epilogue != "none"
    return (
        DEPTH_ACT * bm * bk * geom.a_tile_bytes
        + DEPTH_WEIGHT * bk * bn * geom.b_tile_bytes
        + (bm * bn * geom.p_tile_bytes if partials else 0)
        + DEPTH_OUT * sb_h * geom.out_block_n(bn) * geom.o_tile_bytes
        + bn * geom.row_tile_bytes  # cb_bias_row / cb_gate_row: one N-block row
        + DEPTH_RESIDUAL * sb_h * bn * geom.r_tile_bytes  # cb_residual_tiles
    )


def _block_k_max(geom, bm, bn, budget):
    """Largest block_k with F(bm, bn, block_k, partials=True) <= budget (F is affine in block_k)."""
    fixed = _footprint(geom, bm, bn, 0, partials=True)
    per_k = DEPTH_ACT * bm * geom.a_tile_bytes + DEPTH_WEIGHT * bn * geom.b_tile_bytes
    if budget < fixed:
        return 0
    return (budget - fixed) // per_k


def _k_split(tk, bk_max):
    """Balanced K split: (block_k_tiles, num_k_blocks, last_block_k_tiles)."""
    bk_max = max(1, min(bk_max, tk))
    num_k_blocks = _cdiv(tk, bk_max)
    block_k = _cdiv(tk, num_k_blocks)
    num_k_blocks = _cdiv(tk, block_k)
    last = tk - (num_k_blocks - 1) * block_k
    assert 0 < last <= block_k
    return block_k, num_k_blocks, last


def _make_blocking(geom, block_b, bm, bn, bk_max):
    block_k, nkb, last = _k_split(geom.tensor_k_tiles, min(bk_max, geom.block_k_cap))
    sb_h, sb_w = choose_subblock(bm, bn, geom.dest_limit)
    return Blocking(block_b, bm, bn, block_k, nkb, last, sb_h, sb_w, geom.epilogue)


def choose_blocking(geom: Geometry, grid: GridChoice, budget: int) -> Blocking:
    tk = geom.tensor_k_tiles
    block_b = grid.core_batches  # costs no L1 (the helper's batch loop reuses per-batch CBs)
    cm, cn = grid.core_m_tiles, grid.core_n_tiles
    k_floor = min(tk, BLOCK_K_FLOOR, geom.block_k_cap)
    one_k_block = geom.block_k_cap >= tk  # the precision cap allows K in a single DEST chain

    # S1: whole per-core region, one resident K-block, no accumulator.
    if one_k_block and _footprint(geom, cm, cn, tk, partials=False) <= budget:
        return _make_blocking(geom, block_b, cm, cn, tk)

    # S2: whole region, K-walk with the accumulator.
    bk_max = _block_k_max(geom, cm, cn, budget)
    if bk_max >= k_floor:
        return _make_blocking(geom, block_b, cm, cn, bk_max)

    # S3: shrink the output block; minimize A/B re-reads, ties to the larger block.
    best_key, best = None, None
    seen = set()
    for nb_m in range(1, cm + 1):
        bm = _cdiv(cm, nb_m)
        for nb_n in range(1, _cdiv(cn, grid.q_n) + 1):
            bn = _round_up(_cdiv(cn, nb_n), grid.q_n)
            if (bm, bn) in seen:
                continue
            seen.add((bm, bn))
            if _footprint(geom, bm, bn, k_floor, partials=True) > budget:
                continue
            eff_nb_m, eff_nb_n = _cdiv(cm, bm), _cdiv(cn, bn)
            reread = tk * (cm * geom.a_tile_bytes * (eff_nb_n - 1) + cn * geom.b_tile_bytes * (eff_nb_m - 1))
            key = (reread, -(bm * bn))
            if best_key is None or key < best_key:
                best_key, best = key, (bm, bn)
    if best is not None:
        bm, bn = best
        if one_k_block and _footprint(geom, bm, bn, tk, partials=False) <= budget:
            return _make_blocking(geom, block_b, bm, bn, tk)
        return _make_blocking(geom, block_b, bm, bn, _block_k_max(geom, bm, bn, budget))

    # S4
    raise RuntimeError(f"genify_matmul: even the minimum block does not fit the L1 CB budget of {budget} bytes")


def _l1_tensor_bytes_per_core(tensors, num_cores):
    total = 0
    for t in tensors:
        mc = t.memory_config()
        if mc.buffer_type != ttnn.BufferType.L1:
            continue
        if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
            total += _cdiv(t.buffer_num_pages(), num_cores) * t.buffer_aligned_page_size()
        else:
            shape = mc.shard_spec.shape
            total += _cdiv(shape[0], TILE) * _cdiv(shape[1], TILE) * t.buffer_aligned_page_size()
    return total


def _lowest_allocated_l1_offset(view):
    """Offset (from the bank base, where the CB region starts) of the lowest allocated L1 block, or
    the bank size when nothing is allocated. L1 tensors are placed top-down but freed blocks leave
    holes, so this -- not `total - allocated` -- is where the CB region must end (program.cpp throws
    "circular buffers ... clash with L1 buffers" otherwise)."""
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else view.total_bytes_per_bank


def l1_cb_budget(device, op_tensors):
    """Bytes per core the CBs may use: the worker-L1 region from l1_unreserved_base (where CBs start
    and where program.cpp bounds them) up to the lowest allocated L1 block, never more than the bank
    minus this op's own L1-resident tensors per core, minus the reserve knob. Note:
    ttnn.get_max_worker_l1_unreserved_size() counts from the kernel-config ring buffer, below the CB
    base, and over-states the budget."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    room = min(
        _lowest_allocated_l1_offset(view),
        view.total_bytes_per_bank - _l1_tensor_bytes_per_core(op_tensors, view.num_banks),
    )
    return room - L1_RESERVE_BYTES


def plan(a, b, out, fp32_dest_acc_en, device, epilogue="none", row=None, residual=None):
    """The full blocking decision (geometry, grid, blocking). Exposed for tests / perf tooling.
    `row` is the bias (bias / bias_relu) or gate (addcmul) tensor, `residual` the addcmul residual."""
    geom = _geometry(a, b, out, fp32_dest_acc_en, epilogue, row, residual)
    grid_size = device.compute_with_storage_grid_size()
    gx, gy = grid_size.x, grid_size.y
    budget = l1_cb_budget(device, [t for t in (a, b, out, row, residual) if t is not None])
    grid, blocking = choose_split(geom, gx, gy, budget)
    return geom, grid, blocking, (gx, gy)


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(a, b, out, compute_config, epilogue="none", row=None, residual=None):
    """`row` = bias (bias / bias_relu) or gate (addcmul); `residual` = addcmul residual."""
    device = a.device()
    geom, grid, blk, (gx, gy) = plan(
        a, b, out, bool(compute_config.fp32_dest_acc_en), device, epilogue=epilogue, row=row, residual=residual
    )
    cores, core_ranges = _logical_cores(grid, gx)

    out_block_n_tiles = geom.out_block_n(blk.block_n_tiles)  # swiglu halves the output width
    cb_row = CB_GATE_ROW if epilogue == "addcmul" else CB_BIAS_ROW

    # ---- CBs: every page count derives from the block / depth knobs ----
    cbs = [
        _cb(CB_ACT_TILES, a.dtype, geom.a_tile_bytes, DEPTH_ACT * blk.block_m_tiles * blk.block_k_tiles, core_ranges),
        _cb(
            CB_WEIGHT_TILES,
            b.dtype,
            geom.b_tile_bytes,
            DEPTH_WEIGHT * blk.block_k_tiles * blk.block_n_tiles,
            core_ranges,
        ),
        _cb(CB_OUTPUT_TILES, out.dtype, geom.o_tile_bytes, DEPTH_OUT * blk.sb_h * out_block_n_tiles, core_ranges),
    ]
    if blk.use_partials:
        # Exactly one block (mechanism cap: in-place packer L1 accumulation).
        cbs.append(
            _cb(CB_PARTIALS, geom.p_dtype, geom.p_tile_bytes, blk.block_m_tiles * blk.block_n_tiles, core_ranges)
        )
    if row is not None:
        # One N-block row of the bias / gate, refreshed once per block.
        cbs.append(_cb(cb_row, row.dtype, geom.row_tile_bytes, blk.block_n_tiles, core_ranges))
    if residual is not None:
        cbs.append(
            _cb(
                CB_RESIDUAL_TILES,
                residual.dtype,
                geom.r_tile_bytes,
                DEPTH_RESIDUAL * blk.sb_h * blk.block_n_tiles,
                core_ranges,
            )
        )

    # Absent epilogue tensors: B stands in for the accessor CT args (never dereferenced).
    row_t = row if row is not None else b
    res_t = residual if residual is not None else b

    # ---- Kernels ----
    reader_ct = [
        CB_ACT_TILES,
        CB_WEIGHT_TILES,
        blk.block_batches,
        blk.block_m_tiles,
        blk.block_n_tiles,
        blk.block_k_tiles,
        blk.num_k_blocks,
        blk.last_block_k_tiles,
        geom.a_tile_bytes,
        geom.b_tile_bytes,
        a.buffer_aligned_page_size(),
        b.buffer_aligned_page_size(),
        EPILOGUES[epilogue],
        cb_row,
        CB_RESIDUAL_TILES,
        blk.sb_h,
        ttnn.tile_size(row_t.dtype),
        row_t.buffer_aligned_page_size(),
        ttnn.tile_size(res_t.dtype),
        res_t.buffer_aligned_page_size(),
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(row_t).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(res_t).get_compile_time_args())

    writer_ct = [
        CB_OUTPUT_TILES,
        blk.block_batches,
        blk.block_m_tiles,
        out_block_n_tiles,
        blk.sb_h,
        geom.o_tile_bytes,
        out.buffer_aligned_page_size(),
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    compute_ct = [
        CB_ACT_TILES,
        CB_WEIGHT_TILES,
        CB_OUTPUT_TILES,
        CB_PARTIALS,
        blk.block_batches,
        blk.block_m_tiles // blk.sb_h,  # in0_num_subblocks
        blk.block_n_tiles // blk.sb_w,  # in1_num_subblocks
        blk.sb_h,
        blk.sb_w,
        blk.block_k_tiles,
        blk.num_k_blocks,
        blk.last_block_k_tiles,
        blk.block_n_tiles,
        EPILOGUES[epilogue],
        cb_row,
        CB_RESIDUAL_TILES,
        blk.block_m_tiles,
        EPILOGUE_BLOCK_TILES,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    a_addr, b_addr, out_addr = a.buffer_address(), b.buffer_address(), out.buffer_address()
    row_addr = row.buffer_address() if row is not None else 0
    res_addr = residual.buffer_address() if residual is not None else 0
    out_div = 2 if geom.swiglu else 1  # B columns per output column
    for bi, i, j, core in cores:
        b0 = bi * grid.core_batches
        batches_valid = min(grid.core_batches, geom.tensor_batches - b0)
        row0 = i * grid.core_m_tiles
        rows_valid = min(grid.core_m_tiles, geom.tensor_m_tiles - row0)
        col0 = j * grid.core_n_tiles
        cols_valid = min(grid.core_n_tiles, geom.tensor_n_tiles - col0)
        nb_b = _cdiv(batches_valid, blk.block_batches)
        nb_m = _cdiv(rows_valid, blk.block_m_tiles)
        nb_n = _cdiv(cols_valid, blk.block_n_tiles)
        region = [b0, batches_valid, row0, rows_valid, col0, cols_valid, nb_b, nb_m, nb_n]
        reader_rt[core.x][core.y] = [
            a_addr,
            b_addr,
            *region,
            geom.tensor_m_tiles,
            geom.tensor_k_tiles,
            geom.tensor_n_tiles,
            geom.weight_batch_stride,
            row_addr,
            res_addr,
        ]
        # The writer walks output columns: swiglu halves col0 / cols_valid (both even, q_n >= 2).
        out_region = [b0, batches_valid, row0, rows_valid, col0 // out_div, cols_valid // out_div, nb_b, nb_m, nb_n]
        writer_rt[core.x][core.y] = [out_addr, *out_region, geom.tensor_m_tiles, geom.out_n_tiles]
        compute_rt[core.x][core.y] = [nb_b * nb_m * nb_n]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NoC0
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # BRISC / NoC1
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
