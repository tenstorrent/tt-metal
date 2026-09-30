# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — ProgramDescriptor (regime `flat_stream`, see op_design.md Blocking Model).

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
# Row-weighted work split (Perf 2, core_balance): per-core DRAM service is uneven by logical grid row (the
# first-block reads of the low rows land up to ~40 us later than the high rows'; measured per-core walls at T640
# C7168 bf16 ~210 us on rows 0-1 vs ~175 us on row 9), so the uniform split left the high rows idle at the end.
# Weight 1 + ROW_WEIGHT * (y - y_mid) / (rows - 1) per core; measured (Blackhole p150, 110 cores, bf16, 3 cards):
# T640 C1792 67.3 -> 63.2 us, T1280 C4096 245.4 -> 232.1 us, T1024 C7168 381 -> 346 us, T640 C4096 149 -> 135 us,
# flat at worst (T256 C1792 +1%, T2048 C4096 +0.3%). 0.08 / 0.12 / 0.20 / 0.24 measured worse overall than 0.16.
# Dynamic balancing (NoC-atomic tail claiming, work stealing) and ramped first blocks measured slower.
ROW_WEIGHT = 0.16
# Carve-out (narrow, measured): float32 residual streams with a bfloat16 sublayer keep the uniform split. Measured
# with the Perf-2 kernels (Blackhole p150, 2 runs each, uniform -> weighted): X fp32 / F bf16 T1000 C7168 679.5 ->
# 705.1 us (+3.8%; +9.4% in the core_balance session), T1000 C1792 178.1 -> 185.9 us (+4.4%). The same dtype pair
# gains at T640 (C1792 -6.8%, C7168 -1.8%), but no predicate separates those cells from the regressions, so the
# pair keeps the uniform split. fp32 / fp32 takes the weighted split (-6.5% T640 C1792, -1.3..-1.6% C7168, T1000
# C1792 +0.2% flat), and so does X bf16 / F fp32 (T640 C7168 264.9 -> 260.1 us).
ROW_WEIGHT_MIXED_FP32_STREAMS = 0.0
NUM_CIRCULAR_BUFFERS = 64  # length of ComputeConfigDescriptor.unpack_to_dest_mode

# The writer loads segment s+1's set while compute still holds segment s's (eager look-ahead): two sets in flight.
assert COEF_DEPTH >= 2, "writer-side expansion needs COEF_DEPTH >= 2"
assert BLOCK_TOKEN_TILES == 1, "flat_stream realizes block_token_tiles through segments; only 1 is built"


def _tensor_token_tiles(shape) -> int:
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * math.ceil(shape[-2] / TILE_HW)


def _work_assignment(grid_size, total_units, row_weight):
    """Contiguous flattened split over the cores `split_work_to_cores(..., row_wise=True)` selects, in split order:
    [(core, start_unit, num_units), ...]. Core i gets 1 unit plus its largest-remainder share of the other
    total - num_cores units, weighted w = 1 + row_weight * (y - y_mid) / (rows - 1) by its logical grid row y
    (see ROW_WEIGHT). row_weight = 0 is the uniform split. Every core keeps >= 1 unit (the kernels need a
    non-empty range)."""
    (_, all_cores, core_group_1, core_group_2, _, _) = ttnn.split_work_to_cores(grid_size, total_units, row_wise=True)
    cores = []
    for group in (core_group_1, core_group_2):
        cores.extend(ttnn.corerange_to_cores(group, None, True))
    rows = grid_size.y
    weights = [1.0 + row_weight * (c.y - (rows - 1) / 2.0) / max(1, rows - 1) for c in cores]
    extra = total_units - len(cores)
    exact = [extra * w / sum(weights) for w in weights]
    counts = [math.floor(e) for e in exact]
    by_remainder = sorted(range(len(cores)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in by_remainder[: extra - sum(counts)]:
        counts[i] += 1
    assignment, start = [], 0
    for core, count in zip(cores, counts):
        assignment.append((core, start, count + 1))
        start += count + 1
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


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    residual: ttnn.Tensor,
    post: ttnn.Tensor,
    comb: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()

    n = post.shape[-1]
    col_tiles_per_row = input_tensor.shape[-1] // TILE_HW  # Ct
    token_tiles = _tensor_token_tiles(list(input_tensor.padded_shape))
    total_units = token_tiles * col_tiles_per_row
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    num_raw_tiles = post_tiles_per_row + comb_tiles_per_row
    coef_tiles_per_stream = math.ceil((n + 1) / 2)  # two coefficient terms per tile (mhc_post_common.hpp)
    num_coef_tiles = n * coef_tiles_per_stream

    sublayer_page = input_tensor.buffer_page_size()
    residual_page = residual.buffer_page_size()
    output_page = output_tensor.buffer_page_size()
    coef_page = post.buffer_page_size()
    assert comb.buffer_page_size() == coef_page
    assert output_page == residual_page

    # ---- work split + block size ----
    grid_size = device.compute_with_storage_grid_size()
    row_weight = ROW_WEIGHT
    if residual.dtype == ttnn.float32 and input_tensor.dtype == ttnn.bfloat16:
        row_weight = ROW_WEIGHT_MIXED_FP32_STREAMS  # carve-out (measured regression, see ROW_WEIGHT_MIXED_FP32_STREAMS)
    all_cores, assignment = _work_assignment(grid_size, total_units, row_weight)
    block_col_tiles_fit = _block_col_tiles_fit(
        n, sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles
    )
    assert block_col_tiles_fit >= 1, "mhc_post: coefficient set + one column block does not fit L1_BUDGET_BYTES"
    block_col_tiles = min(block_col_tiles_fit, _max_segment_col_tiles(assignment, col_tiles_per_row))
    max_units_per_core = max(count for _, _, count in assignment)
    block_col_tiles = min(block_col_tiles, math.ceil(max_units_per_core / MIN_BLOCKS_PER_CORE))
    if MAX_BLOCK_COL_TILES is not None:
        block_col_tiles = min(block_col_tiles, MAX_BLOCK_COL_TILES)
    block_col_tiles = max(1, block_col_tiles)

    # ---- circular buffers ----
    cbs = [
        _cb(CB_SUBLAYER_TILES, input_tensor.dtype, sublayer_page, DEPTH_IN * block_col_tiles, all_cores),
        _cb(CB_RESIDUAL_TILES, residual.dtype, residual_page, DEPTH_IN * n * block_col_tiles, all_cores),
        _cb(CB_COEF_RAW, post.dtype, coef_page, num_raw_tiles, all_cores),
        _cb(CB_COEF_BCAST, post.dtype, coef_page, COEF_DEPTH * num_coef_tiles, all_cores),
        _cb(CB_OUTPUT_TILES, output_tensor.dtype, output_page, DEPTH_OUT * n * block_col_tiles, all_cores),
    ]

    # ---- read help (see HELP_MIN_BLOCKS) ----
    max_blocks_per_core = max(
        _core_blocks(start, count, col_tiles_per_row, block_col_tiles) for _, start, count in assignment
    )
    read_help = HELP_MIN_BLOCKS is not None and max_blocks_per_core >= HELP_MIN_BLOCKS
    if residual.dtype == ttnn.float32 and not HELP_FP32_STREAMS:
        read_help = False  # carve-out: compute-bound datapath (see HELP_FP32_STREAMS)

    # ---- data movement: one source (mhc_post_dm.cpp), role 0 = reader (NCRISC), role 1 = writer (BRISC) ----
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
        ]
        for t in (input_tensor, residual, output_tensor, post, comb):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return ct

    # ---- compute ----
    compute_ct = [
        n,
        col_tiles_per_row,
        block_col_tiles,
        CB_SUBLAYER_TILES,
        CB_RESIDUAL_TILES,
        CB_COEF_BCAST,
        CB_OUTPUT_TILES,
        coef_tiles_per_stream,
    ]

    dm_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    f_addr, x_addr = input_tensor.buffer_address(), residual.buffer_address()
    p_addr, m_addr = post.buffer_address(), comb.buffer_address()
    o_addr = output_tensor.buffer_address()
    for core, start, count in assignment:
        dm_rt[core.x][core.y] = [f_addr, x_addr, o_addr, p_addr, m_addr, start, count]
        compute_rt[core.x][core.y] = [start, count]

    # UnpackToDestFp32 on every Float32 CB compute reads with copy_tile (derived from the CB format).
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
        ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_cores, initial_value=0) for sem in (SEM_RD_GO, SEM_RD_DONE)
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
