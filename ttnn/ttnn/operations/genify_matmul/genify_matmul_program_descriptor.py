# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""genify_matmul program descriptor — regime R1 `interleaved_out_split` (op_design.md, Blocking Model).

Host side of the blocking scheme:
  * choose_grid()     — the core-assignment knob: logical (grid_b, grid_m, grid_n) factorization,
                        lexicographic min of (per-core output tiles W, per-core read bytes R, cores).
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
CB_OUTPUT_TILES = 16  # output row groups, compute -> writer
CB_PARTIALS = 24  # K-accumulator (packer L1 acc), compute -> compute; only when num_k_blocks > 1

# ---- Buffer-depth knobs ----
DEPTH_ACT = 2  # A K-blocks in flight
DEPTH_WEIGHT = 2  # B K-blocks in flight
DEPTH_OUT = 2  # output row groups in flight

# ---- Blocking knobs ----
BLOCK_K_FLOOR = 8  # smallest K-block the solve accepts (clamped to tensor_k_tiles)
Q_N_MAX = 4  # subblock quantum on core_n_tiles (a 4-tile subblock always exists)
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

    @property
    def use_partials(self):
        # Epilogue none: the K-accumulator exists only for a multi-K-block walk.
        return self.num_k_blocks > 1


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _geometry(a, b, out, fp32_dest_acc_en):
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


def choose_grid(geom: Geometry, gx: int, gy: int) -> GridChoice:
    C = gx * gy
    tb, tm, tn, tk = geom.tensor_batches, geom.tensor_m_tiles, geom.tensor_n_tiles, geom.tensor_k_tiles
    best_key, best = None, None
    for gb in range(1, min(tb, C) + 1):
        for gm in range(1, min(tm, C // gb) + 1):
            for gn in range(1, min(tn, C // (gb * gm)) + 1):
                core_b = _cdiv(tb, gb)
                core_m = _cdiv(tm, gm)
                n_per = _cdiv(tn, gn)
                q_n = min(Q_N_MAX, _pow2_floor(n_per))
                core_n = _round_up(n_per, q_n)
                ub, um, un = _cdiv(tb, core_b), _cdiv(tm, core_m), _cdiv(tn, core_n)
                placement = _placement_for(ub, um, un, gx, gy)
                if placement is None:
                    continue
                W = core_b * core_m * core_n
                R = core_b * tk * (core_m * geom.a_tile_bytes + core_n * geom.b_tile_bytes)
                key = (W, R, ub * um * un)
                if best_key is None or key < best_key:
                    best_key = key
                    best = GridChoice(ub, um, un, core_b, core_m, core_n, q_n, placement)
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
    sb_h, _ = choose_subblock(bm, bn, geom.dest_limit)
    return (
        DEPTH_ACT * bm * bk * geom.a_tile_bytes
        + DEPTH_WEIGHT * bk * bn * geom.b_tile_bytes
        + (bm * bn * geom.p_tile_bytes if partials else 0)
        + DEPTH_OUT * sb_h * bn * geom.o_tile_bytes
    )


def _block_k_max(geom, bm, bn, budget):
    """Largest block_k with F(bm, bn, block_k, partials=True) <= budget (F is affine in block_k)."""
    sb_h, _ = choose_subblock(bm, bn, geom.dest_limit)
    fixed = bm * bn * geom.p_tile_bytes + DEPTH_OUT * sb_h * bn * geom.o_tile_bytes
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
    block_k, nkb, last = _k_split(geom.tensor_k_tiles, bk_max)
    sb_h, sb_w = choose_subblock(bm, bn, geom.dest_limit)
    return Blocking(block_b, bm, bn, block_k, nkb, last, sb_h, sb_w)


def choose_blocking(geom: Geometry, grid: GridChoice, budget: int) -> Blocking:
    tk = geom.tensor_k_tiles
    block_b = grid.core_batches  # costs no L1 (the helper's batch loop reuses per-batch CBs)
    cm, cn = grid.core_m_tiles, grid.core_n_tiles
    k_floor = min(tk, BLOCK_K_FLOOR)

    # S1: whole per-core region, one resident K-block, no accumulator.
    if _footprint(geom, cm, cn, tk, partials=False) <= budget:
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
        if _footprint(geom, bm, bn, tk, partials=False) <= budget:
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


def l1_cb_budget(device, op_tensors):
    """Bytes per core the CBs may use: the worker-L1 region above l1_unreserved_base (where CBs start
    and where program.cpp bounds them), minus this op's L1-resident tensors per core, minus the
    reserve knob. The allocator's live per-bank usage is a floor on the tensor term (it also sees
    foreign L1 tensors). Note: ttnn.get_max_worker_l1_unreserved_size() counts from the kernel-config
    ring buffer, below the CB base, and over-states the budget."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    tensor_bytes = max(
        _l1_tensor_bytes_per_core(op_tensors, view.num_banks),
        view.total_bytes_allocated_per_bank,
    )
    return view.total_bytes_per_bank - tensor_bytes - L1_RESERVE_BYTES


def plan(a, b, out, fp32_dest_acc_en, device):
    """The full blocking decision (geometry, grid, blocking). Exposed for tests / perf tooling."""
    geom = _geometry(a, b, out, fp32_dest_acc_en)
    grid_size = device.compute_with_storage_grid_size()
    gx, gy = grid_size.x, grid_size.y
    grid = choose_grid(geom, gx, gy)
    budget = l1_cb_budget(device, [a, b, out])
    blocking = choose_blocking(geom, grid, budget)
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


def create_program_descriptor(a, b, out, compute_config) -> ttnn.ProgramDescriptor:
    device = a.device()
    geom, grid, blk, (gx, gy) = plan(a, b, out, bool(compute_config.fp32_dest_acc_en), device)
    cores, core_ranges = _logical_cores(grid, gx)

    out_block_n_tiles = blk.block_n_tiles  # epilogue none: output width == B width

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
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())

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
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    a_addr, b_addr, out_addr = a.buffer_address(), b.buffer_address(), out.buffer_address()
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
        ]
        writer_rt[core.x][core.y] = [out_addr, *region, geom.tensor_m_tiles, geom.out_n_tiles]
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
