# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — ProgramDescriptor (Phase 0: regime R1 plain-2D).

Work split (op_design.md "Split chooser"): the worker list L is the max_grid
rectangle in row-major order. R row groups x Q column members; row group g
is L[g*Q:(g+1)*Q] and owns a contiguous range of token-tile rows, member q a
contiguous range of C-column tiles (both ragged +-1). A core's block is one
token-tile row (BLOCK_ROWS) x block_cols column tiles x n streams.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth; every CB size / trip count derives from these)
# ---------------------------------------------------------------------------
BLOCK_ROWS = 1  # token-tile rows per block (reasoned decision, op_design.md "Why block_rows = 1")
IN_DEPTH = 2  # units in flight in cb_sublayer_tiles / cb_stream_tiles
OUT_DEPTH = 2  # units in flight in cb_mixed_out_tiles
COEF_DEPTH = 2  # token rows of expanded coefficients in flight
BLOCK_L1_BUDGET = 384 * 1024  # bytes available to block-scaled CBs (ledger symbol)

# Split-chooser model constants (catalog double_buffer measurements, Blackhole)
DRAM_BW = 512e9
CORE_BW = 17.9e9

# CB indices (any distinct values)
CB_SUBLAYER_TILES = 0  # F tiles, one per unit
CB_STREAM_TILES = 1  # X tiles, n per unit
CB_COEF_RAW = 2  # reader-private landing of the raw post / comb tile rows
CB_COEF_COLS = 3  # coefficients expanded to column 0 of their own page (UnpackToDestFp32)
CB_MIXED_OUT_TILES = 4  # X' tiles, n per unit

NUM_CBS = 64


def _split(total, parts):
    """Ragged +-1 split: the first `total % parts` parts take one more. Returns [(start, count)]."""
    base, extra = divmod(total, parts)
    out, start = [], 0
    for p in range(parts):
        cnt = base + (1 if p < extra else 0)
        out.append((start, cnt))
        start += cnt
    return out


def worker_list(device, max_grid):
    """L: the max_grid rectangle, row-major. (Port reservation for an active fabric is the feed refinement's.)"""
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    if max_grid is not None:
        gx, gy = min(gx, max_grid.x), min(gy, max_grid.y)
    return [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx)], gx


def choose_split(t_tiles, c_tiles, num_workers, unit_bytes, has_w, n, C):
    """Exact chooser of op_design.md: minimise max(total DRAM / DRAM_BW, max core bytes / CORE_BW), ties -> smaller R."""
    best = None
    for R in range(1, min(t_tiles, num_workers) + 1):
        Q = min(num_workers // R, c_tiles)
        if Q < 1:
            break
        total = t_tiles * c_tiles * unit_bytes + has_w * R * n * C * 128
        max_core = -(-t_tiles // R) * -(-c_tiles // Q) * unit_bytes + has_w * n * -(-c_tiles // Q) * 4096
        est = max(total / DRAM_BW, max_core / CORE_BW)
        if best is None or est < best[0]:
            best = (est, R, Q)
    return best[1], best[2]


def choose_block_cols(core_c_tiles, bytes_per_block_col):
    cap = max(1, BLOCK_L1_BUDGET // max(1, bytes_per_block_col))
    return max(1, min(core_c_tiles, cap))


def _merged_ranges(cores, gx):
    """Row-major prefix of the rectangle -> at most two rectangles (full rows + partial last row)."""
    num = len(cores)
    full_rows, rem = divmod(num, gx)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(
    f_tensor,
    x_tensor,
    post_tensor,
    comb_tensor,
    out_tensor,
    *,
    n,
    post_col,
    comb_col,
    math_fidelity,
    math_approx_mode,
    max_grid,
    mode,
):
    assert mode == "plain", mode
    device = x_tensor.device()

    # ---------------- geometry ----------------
    f_pad = list(f_tensor.padded_shape)
    t_tiles = 1
    for d in f_pad[:-1]:
        t_tiles *= d
    t_tiles //= TILE  # tensor_t_tiles: every leading dim folds into whole (padded) images
    c_tiles = f_pad[-1] // TILE  # tensor_c_tiles
    C = f_pad[-1]
    post_wt = post_tensor.padded_shape[-1] // TILE
    comb_wt = comb_tensor.padded_shape[-1] // TILE

    f_page = f_tensor.buffer_page_size()
    x_page = x_tensor.buffer_page_size()
    out_page = out_tensor.buffer_page_size()
    coef_page = post_tensor.buffer_page_size()  # fp32 tile
    n_coef = n + n * n

    # ---------------- work split ----------------
    workers, gx = worker_list(device, max_grid)
    has_w = 0  # plain mode: no W (the score must not see mode beyond has_W)
    unit_bytes = f_page + 2 * n * x_page
    R, Q = choose_split(t_tiles, c_tiles, len(workers), unit_bytes, has_w, n, C)
    row_ranges = _split(t_tiles, R)
    col_ranges = _split(c_tiles, Q)
    used = workers[: R * Q]
    core_ranges = _merged_ranges(used, gx)
    max_core_c_tiles = col_ranges[0][1]
    # block-scaled bytes per column tile: plain mode streams every CB over c, so only the loop extent scales
    block_cols = choose_block_cols(max_core_c_tiles, n * 2048)

    # ---------------- CBs ----------------
    cbs = [
        _cb(CB_SUBLAYER_TILES, f_tensor.dtype, f_page, IN_DEPTH, core_ranges),
        _cb(CB_STREAM_TILES, x_tensor.dtype, x_page, n * IN_DEPTH, core_ranges),
        _cb(CB_COEF_RAW, ttnn.float32, coef_page, 2, core_ranges),  # post + comb (1 used when one tensor)
        _cb(CB_COEF_COLS, ttnn.float32, coef_page, n_coef * COEF_DEPTH, core_ranges),
        _cb(CB_MIXED_OUT_TILES, out_tensor.dtype, out_page, n * OUT_DEPTH, core_ranges),
    ]

    # ---------------- per-core runtime args ----------------
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for g in range(R):
        t0, tn = row_ranges[g]
        for q in range(Q):
            c0, cn = col_ranges[q]
            core = used[g * Q + q]
            reader_rt[core.x][core.y] = [t0, tn, c0, cn]
            writer_rt[core.x][core.y] = [t0, tn, c0, cn]
            compute_rt[core.x][core.y] = [tn, cn]

    # ---------------- kernels ----------------
    reader_ct = [
        CB_SUBLAYER_TILES,
        CB_STREAM_TILES,
        CB_COEF_RAW,
        CB_COEF_COLS,
        n,
        BLOCK_ROWS,
        block_cols,
        c_tiles,
        post_wt,
        comb_wt,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(f_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(x_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(post_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(comb_tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[
            f_tensor.buffer_address(),
            x_tensor.buffer_address(),
            post_tensor.buffer_address(),
            comb_tensor.buffer_address(),
            post_col,
            comb_col,
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct = [CB_MIXED_OUT_TILES, n, BLOCK_ROWS, block_cols, c_tiles]
    writer_ct.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[out_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
    # Every fp32 CB the compute reads goes to DEST unpacked (never through tf32 srcA/srcB).
    unpack_modes[CB_COEF_COLS] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if f_tensor.dtype == ttnn.float32:
        unpack_modes[CB_SUBLAYER_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if x_tensor.dtype == ttnn.float32:
        unpack_modes[CB_STREAM_TILES] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=math_approx_mode,
    )
    compute_cfg.unpack_to_dest_mode = unpack_modes
    compute_ct = [CB_SUBLAYER_TILES, CB_STREAM_TILES, CB_COEF_COLS, CB_MIXED_OUT_TILES, n, BLOCK_ROWS, block_cols]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_cfg,
    )

    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
