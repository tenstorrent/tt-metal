# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""core_balance candidate: mhc_post ProgramDescriptor + dynamic tail pool.

Original doc: mhc_post — ProgramDescriptor (regime `flat_stream`, see op_design.md Blocking Model).

Flattened (token-tile row r, column tile c) units, r-major, split contiguously over the full compute
grid (`split_work_to_cores(..., row_wise=True)`). Per core: segments (one token row each), blocks of
`block_col_tiles` columns. Every block knob below is defined once; CB sizes, CT/RT args derive from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HW = 32

# ---- CB slots (semantic names) ----
CB_SUBLAYER_TILES = 0  # F block tiles              reader (+ writer's read help) -> compute
CB_RESIDUAL_TILES = 1  # X block tiles (n streams)  reader  -> compute
CB_COEF_RAW = 2  # raw post / comb tiles of one token row, private scratch of the writer (coefficient expander)
CB_COEF_BCAST = 3  # n * P half-packed column-broadcast fp32 coefficient tiles, writer -> compute
CB_OUTPUT_TILES = 16  # X' block tiles (n streams) compute -> writer

# ---- Block-model knobs (single source of truth) ----
BLOCK_TOKEN_TILES = 1  # a block never spans two token rows (realized by segments)
DEPTH_IN = 2  # blocks in flight on cb_sublayer_tiles / cb_residual_tiles
DEPTH_OUT = 2  # blocks in flight on cb_output_tiles
COEF_DEPTH = 2  # token-row coefficient sets in flight on cb_coef_bcast
L1_BUDGET_BYTES = 1 << 20  # CB budget per core
# Block-size policy: block_col_tiles = min(L1 fit, longest segment, MAX_BLOCK_COL_TILES,
# ceil(max units per core / MIN_BLOCKS_PER_CORE)). The coarsest L1 fit leaves a core with 1-2 blocks, so read,
# mix and write barely overlap and every core bursts its whole prefetch at DRAM at once. Measured (Blackhole,
# 110 cores, bf16): cap 8 is the best cap at T640 C7168 / T1280 C4096, and >= 3 blocks per core is best at
# T640 C1792 (10-11 units per core -> B = 4). Deeper DEPTH_IN (3) measured slower.
MAX_BLOCK_COL_TILES = 8  # cap on block_col_tiles (None = no cap)
MIN_BLOCKS_PER_CORE = 3  # blocks the busiest core's range is cut into, at least (1 = no constraint)
# DEST sync mode of the compute kernel (overrides the caller's dst_full_sync_en; an internal schedule
# choice). SyncFull gives 8 fp32 DEST slots, enough for one output column's P coefficient tiles + n+1
# data tiles at n = 4; the kernel derives its window layout from DEST_AUTO_LIMIT. False (SyncHalf, 4
# slots) compiles only for n <= 2 (the kernel static_asserts the fit).
DST_FULL_SYNC = True
# Read help (Perf 1, split_noc_reads; mhc_post_dm.cpp): the writer RISC (BRISC) reads the F block of every block
# index >= HELP_FROM_BLOCK straight into the reader's window, on NoC0 (dynamic-NoC mode). Enabled when the busiest
# core has >= HELP_MIN_BLOCKS blocks; measured (Blackhole p150, 110 cores, bf16): -3..-11% at >= 6 blocks per core
# (T1280 C4096 268 -> 246 us, T1024 C5120 275 -> 245 us), +4..+18% at 3-5 blocks (T640 C1792 67 -> 80 us,
# T2048 C1792 +4%), which is why fewer blocks keep the unhelped schedule (the same kernel, help compiled out).
# HELP_MIN_BLOCKS = None disables the help everywhere; 1 enables it at every block count.
HELP_MIN_BLOCKS = 6
# Carve-out: float32 residual streams keep the unhelped schedule. That datapath is compute-bound (fp32 X' through
# UnpackToDestFp32), so the help cannot shorten the critical path and only adds its hand-off; measured
# T1000 C7168 X fp32 / F bf16 677 -> 700 us (+3.2%, 3 runs, spread < 1%), X fp32 / F fp32 +1.7%, other fp32-X
# cells +-2%. True applies the help to fp32 streams too (test coverage).
HELP_FP32_STREAMS = False
HELP_FROM_BLOCK = 1  # help from block 0 measured slower (it delays the writer's first coefficient set)
SEM_RD_GO = 0  # reader -> helper: window of block k reserved (monotonic block counter)
SEM_RD_DONE = 1  # helper -> reader: F of block k landed
NUM_CIRCULAR_BUFFERS = 64  # length of ComputeConfigDescriptor.unpack_to_dest_mode
CB_META_C = 4  # dynamic pool: claimed chunk ids, reader -> compute (one uint32 per chunk, 0 = end)
CB_META_W = 5  # dynamic pool: claimed chunk ids, reader -> writer
SEM_CLAIM = 2  # global chunk counter (only the copy on the counter core is used)
SEM_RET = 3  # local landing slot of the atomic's returned old value
META_PAGE = 16
META_DEPTH = 4
CB_SCAN = 6  # pool_mode 2: reader-private scratch, one 16 B slot per queue (counter scan)


from dataclasses import dataclass, field


@dataclass
class BalanceConfig:
    pool_frac: float = 0.0  # mode 1: fraction of all units in the global pool; mode 2: fraction of each core's range
    pool_mode: int = 1  # 1 = one global tail pool, 2 = per-core tail queues (owner first, then stealing)
    help_dynamic: bool = True  # the BRISC read help also covers claimed chunks
    stride: int | None = None  # mode 2 victim stride (coprime with the core count); None = auto
    steal_min: int = 2  # mode 2: a victim queue must have >= this many chunks left
    chunk_cols: int | None = None  # columns per claimed chunk (<= B); None = B
    coef_depth: int = COEF_DEPTH  # coefficient sets in flight
    weights: object = None  # optional callable (device, logical core) -> static weight (position-aware static split)
    min_pool_chunks: int = 1
    claim_ahead: bool = False  # claim chunk d+1 when starting chunk d (coef set lead time)
    ramp0: int | None = None  # first block width of the ramped walk (x2 per block up to B); None = B (op's walk)


# The writer loads segment s+1's set while compute still holds segment s's: two sets in flight.
assert COEF_DEPTH >= 2, "writer-side expansion needs COEF_DEPTH >= 2"
assert BLOCK_TOKEN_TILES == 1, "flat_stream realizes block_token_tiles through segments; only 1 is built"


def _tensor_token_tiles(shape) -> int:
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * math.ceil(shape[-2] / TILE_HW)


def _work_assignment(grid_size, total_units):
    """Contiguous flattened split: [(core, start_unit, num_units), ...] in split order."""
    (
        _,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, total_units, row_wise=True)
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_per_core_g1), (core_group_2, units_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == total_units
    return all_cores, assignment


def _max_segment_col_tiles(assignment, col_tiles_per_row) -> int:
    longest = 0
    for _, start, count in assignment:
        u, left = start, count
        while left:
            seg = min(col_tiles_per_row - u % col_tiles_per_row, left)
            longest = max(longest, seg)
            u += seg
            left -= seg
    return longest


def _core_blocks(start, count, col_tiles_per_row, block_col_tiles) -> int:
    """Blocks of one core's unit range (segments of one token row, each cut into blocks) — kernel derivation."""
    blocks, u, left = 0, start, count
    while left:
        seg = min(col_tiles_per_row - u % col_tiles_per_row, left)
        blocks += math.ceil(seg / block_col_tiles)
        u += seg
        left -= seg
    return blocks


def _block_col_tiles_fit(
    n, sublayer_tile_bytes, residual_tile_bytes, coef_tile_bytes, num_coef_tiles, num_raw_tiles
) -> int:
    """Closed form from l1_ledger.md: coarsest block that fits L1_BUDGET_BYTES."""
    coef_bytes = COEF_DEPTH * num_coef_tiles * coef_tile_bytes + num_raw_tiles * coef_tile_bytes
    per_col = DEPTH_IN * (sublayer_tile_bytes + n * residual_tile_bytes) + DEPTH_OUT * n * residual_tile_bytes
    return (L1_BUDGET_BYTES - coef_bytes) // per_col


def _cb(index, dtype, page_bytes, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _static_assignment(assignment, static_units, weights, device=None):
    """Re-split [0, static_units) over the SAME cores (split order): uniform (+-1) or by weights(x, y)."""
    cores = [core for core, _, _ in assignment]
    if weights is None:
        base, extra = divmod(static_units, len(cores))
        counts = [base + (1 if i < extra else 0) for i in range(len(cores))]
    else:
        w = [float(weights(device, c)) for c in cores]
        tot = sum(w)
        exact = [static_units * wi / tot for wi in w]
        counts = [int(math.floor(e)) for e in exact]
        rem = static_units - sum(counts)
        order = sorted(range(len(cores)), key=lambda i: exact[i] - counts[i], reverse=True)
        for i in order[:rem]:
            counts[i] += 1
    out, start = [], 0
    for core, cnt in zip(cores, counts):
        out.append((core, start, cnt))
        start += cnt
    assert start == static_units
    return out


def _queue(params, v):
    n1, g1, g2, q = params
    start = v * g1 if v < n1 else n1 * g1 + (v - n1) * g2
    qe = start + (g1 if v < n1 else g2)
    return qe - q, qe


def _queue_chunks(params, v, ct, chunk_cols):
    """Chunks of queue v (kernel mirror: PoolParams::queue_chunks)."""
    qs, qe = _queue(params, v)
    if qe <= qs:
        return 0
    cpr = math.ceil(ct / chunk_cols)
    aligned = lambda u: (u // ct) * cpr + (u % ct) // chunk_cols
    return aligned(qe - 1) - aligned(qs) + 1


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    residual: ttnn.Tensor,
    post: ttnn.Tensor,
    comb: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
    bal: BalanceConfig | None = None,
) -> ttnn.ProgramDescriptor:
    bal = bal or BalanceConfig()
    device = input_tensor.device()

    n = post.shape[-1]
    col_tiles_per_row = input_tensor.shape[-1] // TILE_HW  # Ct
    token_tiles = _tensor_token_tiles(list(input_tensor.padded_shape))
    total_units = token_tiles * col_tiles_per_row
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    num_raw_tiles = post_tiles_per_row + comb_tiles_per_row
    coef_tiles_per_stream = math.ceil((n + 1) / 2)
    num_coef_tiles = n * coef_tiles_per_stream
    coef_depth = bal.coef_depth
    assert coef_depth >= 2

    sublayer_page = input_tensor.buffer_page_size()
    residual_page = residual.buffer_page_size()
    output_page = output_tensor.buffer_page_size()
    coef_page = post.buffer_page_size()
    assert comb.buffer_page_size() == coef_page
    assert output_page == residual_page

    # ---- work split + block size: IDENTICAL policy to the op (full uniform split decides B and the help) ----
    grid_size = device.compute_with_storage_grid_size()
    all_cores, assignment = _work_assignment(grid_size, total_units)
    coef_bytes_extra = (coef_depth - COEF_DEPTH) * num_coef_tiles * coef_page + 2 * META_DEPTH * META_PAGE
    block_col_tiles_fit = _block_col_tiles_fit(
        n, sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles
    )
    per_col = DEPTH_IN * (sublayer_page + n * residual_page) + DEPTH_OUT * n * residual_page
    block_col_tiles_fit -= math.ceil(coef_bytes_extra / per_col) if coef_bytes_extra > 0 else 0
    assert block_col_tiles_fit >= 1
    block_col_tiles = min(block_col_tiles_fit, _max_segment_col_tiles(assignment, col_tiles_per_row))
    max_units_per_core = max(count for _, _, count in assignment)
    block_col_tiles = min(block_col_tiles, math.ceil(max_units_per_core / MIN_BLOCKS_PER_CORE))
    if MAX_BLOCK_COL_TILES is not None:
        block_col_tiles = min(block_col_tiles, MAX_BLOCK_COL_TILES)
    block_col_tiles = max(1, block_col_tiles)

    max_blocks_per_core = max(
        _core_blocks(start, count, col_tiles_per_row, block_col_tiles) for _, start, count in assignment
    )
    read_help = HELP_MIN_BLOCKS is not None and max_blocks_per_core >= HELP_MIN_BLOCKS
    if residual.dtype == ttnn.float32 and not HELP_FP32_STREAMS:
        read_help = False

    # ---- run-time claimed tail queues (see mhc_post_common.hpp PoolParams) ----
    chunk_cols = min(bal.chunk_cols or block_col_tiles, block_col_tiles)
    ramp0 = max(1, min(bal.ramp0 or block_col_tiles, block_col_tiles))
    num_cores = len(assignment)
    pool_mode = 0
    pool_rt = [0, 0, 0, 0, 1]  # n1, g1, g2, q, maxq
    if bal.pool_frac > 0:
        if bal.pool_mode == 1:
            q = int(round(bal.pool_frac * total_units))
            params = (1, total_units, 0, q)
            nqueues = 1
        else:
            counts = [c for _, _, c in assignment]
            g1 = counts[0]
            n1 = sum(1 for c in counts if c == g1)
            g2 = counts[-1] if n1 < len(counts) else g1
            q = min(int(round(bal.pool_frac * g2)), g2)
            params = (n1, g1, g2, q)
            nqueues = num_cores
        chunks = [_queue_chunks(params, v, col_tiles_per_row, chunk_cols) for v in range(nqueues)]
        if q > 0 and sum(chunks) >= bal.min_pool_chunks:
            pool_mode = bal.pool_mode
            pool_rt = list(params) + [max(chunks)]
            if pool_mode == 1:
                assignment = _static_assignment(assignment, total_units - q, bal.weights, device)
            else:
                assignment = [(core, start, count - q) for core, start, count in assignment]
    if pool_mode == 0 and bal.weights is not None:
        assignment = _static_assignment(assignment, total_units, bal.weights, device)
    dynamic = pool_mode != 0
    stride = bal.stride or next(s for s in range(max(1, num_cores // 3), num_cores + 1) if math.gcd(s, num_cores) == 1)

    cbs = [
        _cb(CB_SUBLAYER_TILES, input_tensor.dtype, sublayer_page, DEPTH_IN * block_col_tiles, all_cores),
        _cb(CB_RESIDUAL_TILES, residual.dtype, residual_page, DEPTH_IN * n * block_col_tiles, all_cores),
        _cb(CB_COEF_RAW, post.dtype, coef_page, num_raw_tiles, all_cores),
        _cb(CB_COEF_BCAST, post.dtype, coef_page, coef_depth * num_coef_tiles, all_cores),
        _cb(CB_OUTPUT_TILES, output_tensor.dtype, output_page, DEPTH_OUT * n * block_col_tiles, all_cores),
    ]
    if dynamic:
        cbs.append(_cb(CB_META_C, ttnn.uint32, META_PAGE, META_DEPTH, all_cores))
        cbs.append(_cb(CB_META_W, ttnn.uint32, META_PAGE, META_DEPTH, all_cores))
        if pool_mode == 2:
            cbs.append(_cb(CB_SCAN, ttnn.uint32, 16, num_cores, all_cores))

    def dm_ct(role):
        ct = [
            n,
            col_tiles_per_row,
            block_col_tiles,
            role,
            int(read_help),
            HELP_FROM_BLOCK,
            DEPTH_IN,
            CB_SUBLAYER_TILES,
            CB_RESIDUAL_TILES,
            CB_OUTPUT_TILES,
            CB_COEF_RAW,
            CB_COEF_BCAST,
            post_tiles_per_row,
            comb_tiles_per_row,
            sublayer_page,
            residual_page,
            coef_page,
            coef_tiles_per_stream,
            SEM_RD_GO,
            SEM_RD_DONE,
            TILE_HW,
            pool_mode,
            chunk_cols,
            CB_META_C,
            CB_META_W,
            SEM_CLAIM,
            SEM_RET,
            int(bal.help_dynamic),
            ramp0,
            CB_SCAN,
        ]
        for t in (input_tensor, residual, output_tensor, post, comb):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return ct

    compute_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_BCAST,
        CB_OUTPUT_TILES,
        coef_tiles_per_stream,
        int(dynamic),
        chunk_cols,
        CB_META_C,
        ramp0,
    ]

    dm_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    f_addr, x_addr = input_tensor.buffer_address(), residual.buffer_address()
    p_addr, m_addr = post.buffer_address(), comb.buffer_address()
    o_addr = output_tensor.buffer_address()

    def packed(core, nq):
        v = device.worker_core_from_logical_core(core)
        assert v.x < 256 and v.y < 256 and nq < (1 << 16)
        return (nq << 16) | (v.x << 8) | v.y

    if pool_mode == 1:
        homes = [packed(ttnn.CoreCoord(0, 0), chunks[0])]
    elif pool_mode == 2:
        homes = [packed(core, chunks[v]) for v, (core, _, _) in enumerate(assignment)]
    else:
        homes = []
    for idx, (core, start, count) in enumerate(assignment):
        dm_rt[core.x][core.y] = (
            [f_addr, x_addr, o_addr, p_addr, m_addr, start, count]
            + pool_rt
            + [idx, num_cores, stride | (bal.steal_min << 16)]
            + homes
        )
        compute_rt[core.x][core.y] = [start, count] + pool_rt

    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    for index, dtype in (
        (CB_SUBLAYER_TILES, input_tensor.dtype),
        (CB_RESIDUAL_TILES, residual.dtype),
        (CB_COEF_BCAST, post.dtype),
    ):
        if dtype == ttnn.float32:
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
    )
    compute_cfg.dst_full_sync_en = DST_FULL_SYNC
    compute_cfg.unpack_to_dest_mode = unpack_modes

    noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if read_help else ttnn.NOC_MODE.DM_DEDICATED_NOC
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_dm.cpp"),
        core_ranges=all_cores,
        compile_time_args=dm_ct(0),
        runtime_args=dm_rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0, noc_mode=noc_mode
        ),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_dm.cpp"),
        core_ranges=all_cores,
        compile_time_args=dm_ct(1),
        runtime_args=dm_rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1, noc_mode=noc_mode
        ),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_cfg,
    )
    semaphores = [
        ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_cores, initial_value=0)
        for sem in (SEM_RD_GO, SEM_RD_DONE, SEM_CLAIM, SEM_RET)
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)


def wall_calibration(json_path, shape_key, alpha=1.0):
    """weights(device, core) = (wall of the core in a profiled `orig` run, cycles)^-alpha — a PER-BOARD calibration
    (calib_walls.json: physical "x,y" -> kernel end, from the p3 session on card 0)."""
    import json as _json

    wall = {tuple(int(v) for v in k.split(",")): t for k, t in _json.load(open(json_path))[shape_key].items()}
    # profiler coords are physical; the logical grid is the sorted physical columns / rows (harvesting removed)
    xs = sorted({c[0] for c in wall})
    ys = sorted({c[1] for c in wall})

    def weights(device, core):
        return wall[(xs[core.x], ys[core.y])] ** (-alpha)

    return weights


def row_gradient(beta):
    """Model weight: linear in the logical grid row, weight = 1 + beta * (y - y_mid) / (rows - 1), rows = grid height.
    beta > 0 gives the higher rows (larger y, measured faster DRAM service on Blackhole p150) more work."""

    def weights(device, core):
        rows = device.compute_with_storage_grid_size().y
        return 1.0 + beta * (core.y - (rows - 1) / 2.0) / max(1, rows - 1)

    return weights


def row_weighted_work_assignment(beta):
    """DROP-IN for the op's `_work_assignment(grid_size, total_units)` (mhc_post_program_descriptor.py):
    same cores, same contiguous r-major order, but core i gets 1 + its largest-remainder share of the other
    total - num_cores units by weight w = 1 + beta * (y - y_mid) / (rows - 1) (y = logical grid row).
    Every core keeps >= 1 unit (the kernels assume a non-empty range)."""

    def _work_assignment(grid_size, total_units):
        (_, all_cores, core_group_1, core_group_2, _, _) = ttnn.split_work_to_cores(
            grid_size, total_units, row_wise=True
        )
        cores = []
        for group in (core_group_1, core_group_2):
            cores.extend(ttnn.corerange_to_cores(group, None, True))
        rows = grid_size.y
        w = [1.0 + beta * (c.y - (rows - 1) / 2.0) / max(1, rows - 1) for c in cores]
        extra = total_units - len(cores)
        exact = [extra * wi / sum(w) for wi in w]
        counts = [int(math.floor(e)) for e in exact]
        for i in sorted(range(len(cores)), key=lambda i: exact[i] - counts[i], reverse=True)[: extra - sum(counts)]:
            counts[i] += 1
        assignment, start = [], 0
        for core, cnt in zip(cores, counts):
            assignment.append((core, start, cnt + 1))
            start += cnt + 1
        assert start == total_units
        return all_cores, assignment

    return _work_assignment
