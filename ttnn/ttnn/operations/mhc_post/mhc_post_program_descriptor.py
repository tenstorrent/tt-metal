# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — ProgramDescriptor (Phase 0: plain-2d regime).

Work split (op_design.md → Work Distribution): the (token-tile row × column tile) plane is
block-split 2-D into R row groups × Q column groups by `choose_split`; each core's per-pass
block (`1 row × block_c c-tiles × n streams`) comes from `choose_block_c`. Both are pure
host functions and the single sources of R, Q and block_c. Every depth / budget below is a
single-source host constant feeding both the CB capacities and the kernels' CT args.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Single-source constants (op_design.md → Work Distribution → Constants)
# ---------------------------------------------------------------------------
N_STREAMS = 4  # n = hc_mult (DeepSeek-V4 / V4.1)
TILE = 32
FP32_TILE_BYTES = 4096

L1_CB_BUDGET = 900 * 1024
MIN_PIPELINE_BLOCKS = 4
W_RESIDENT_BUDGET = 384 * 1024  # fused regimes (deferred)
COMBINE_FANIN = 16  # fused regimes (deferred)
COMBINE_DEPTH = 2  # fused regimes (deferred)
COMBINE_LAG = 1  # fused regimes (deferred)
IO_DEPTH = 2  # cb_f_tiles / cb_x_tiles depth, in blocks
OUT_DEPTH = 2  # cb_out_tiles depth, in blocks (design: 1; 2 measured -8..10% on DM-bound shapes)
COEF_DEPTH = 1  # cb_coef_cols depth, in rows
BLOCK_ROWS = 1  # rows per block (see op_design.md "Why block_rows = 1")

# CB indices (semantic names; any distinct values)
CB_COEF_SRC = 0
CB_COEF_COLS = 1
CB_F_TILES = 2
CB_X_TILES = 3
CB_OUT_TILES = 4


def _tile_bytes(dtype):
    return ttnn.tile_size(dtype)


def _n_src(mode, coef_source):
    """Coefficient source pages per row: packed → 1, separate → 2 (+1 separate P in collapse)."""
    if coef_source == "packed":
        return 1
    return 3 if mode == "collapse" else 2


def _n_coef_cols(mode, n=N_STREAMS):
    return n + n * n + (n if mode == "collapse" else 0)


# ---------------------------------------------------------------------------
# choose_split / choose_block_c (host, pure)
# ---------------------------------------------------------------------------


def choose_split(Rt, Ct, *, f_tile, x_tile, n_src, G, fused=False, n=N_STREAMS):
    """(R, Q) minimizing the busiest core's DRAM bytes. Tie-break: smaller R·Q, then smaller Q."""
    unit_bytes = f_tile + 2 * n * x_tile
    row_bytes = n_src * FP32_TILE_BYTES
    w_bytes_per_c = n * FP32_TILE_BYTES if fused else 0
    best = None
    for R in range(1, min(Rt, G) + 1):
        rt_max = -(-Rt // R)
        for Q in range(1, min(Ct, G // R) + 1):
            ct_max = -(-Ct // Q)
            if fused and n * ct_max * FP32_TILE_BYTES > W_RESIDENT_BUDGET:
                continue
            cost = rt_max * (ct_max * unit_bytes + row_bytes) + ct_max * w_bytes_per_c
            key = (cost, R * Q, Q)
            if best is None or key < best[0]:
                best = (key, R, Q)
    return best[1], best[2]


def plain_fixed_bytes(n_src, n_coef_cols):
    return n_src * FP32_TILE_BYTES + n_coef_cols * COEF_DEPTH * FP32_TILE_BYTES


def plain_per_c_bytes(f_tile, x_tile, n=N_STREAMS):
    return IO_DEPTH * f_tile + IO_DEPTH * n * x_tile + OUT_DEPTH * n * x_tile


def choose_block_c(core_c_max, core_rows_max, *, fixed_bytes, per_c_bytes):
    block_c_cap = max(1, (L1_CB_BUDGET - fixed_bytes) // per_c_bytes)
    nb = max(-(-core_c_max // block_c_cap), min(core_c_max, -(-MIN_PIPELINE_BLOCKS // core_rows_max)))
    return -(-core_c_max // nb)


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def _core_ranges(num_cores, gx):
    """Row-wise enumeration k → (k % gx, k // gx), as a CoreRangeSet."""
    ranges = []
    full_rows = num_cores // gx
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
    rem = num_cores % gx
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor,
    residual,
    post,
    comb,
    output_tensor,
    *,
    compute_config,
    post_col=0,
    comb_col=0,
    max_grid=None,
):
    n = N_STREAMS
    mode, coef_source = "plain", "separate"
    x_shape = list(residual.shape)
    T, nC = x_shape[-2], x_shape[-1]
    C = nC // n
    batch = math.prod(x_shape[:-2]) if len(x_shape) > 2 else 1
    Rt = batch * (-(-T // TILE))  # token-tile rows, per image ceil(T/32)
    Ct = C // TILE
    post_wt = -(-list(post.shape)[-1] // TILE)  # tile columns of the post tensor
    comb_wt = -(-list(comb.shape)[-1] // TILE)

    # --- grid (single source G) ---
    device = residual.device()
    dev_grid = device.compute_with_storage_grid_size()
    gx, gy = dev_grid.x, dev_grid.y
    if max_grid is not None:
        gx, gy = max(1, min(max_grid.x, gx)), max(1, min(max_grid.y, gy))
    G = gx * gy

    f_tile = _tile_bytes(input_tensor.dtype)
    x_tile = _tile_bytes(residual.dtype)
    n_src = _n_src(mode, coef_source)
    n_coef_cols = _n_coef_cols(mode)

    R, Q = choose_split(Rt, Ct, f_tile=f_tile, x_tile=x_tile, n_src=n_src, G=G, fused=(mode != "plain"))
    core_rows_max = -(-Rt // R)
    core_c_max = -(-Ct // Q)
    block_c = choose_block_c(
        core_c_max,
        core_rows_max,
        fixed_bytes=plain_fixed_bytes(n_src, n_coef_cols),
        per_c_bytes=plain_per_c_bytes(f_tile, x_tile),
    )
    num_cores = R * Q
    core_ranges = _core_ranges(num_cores, gx)

    # --- circular buffers (capacities derive from the knobs above) ---
    cbs = [
        _cb(CB_COEF_SRC, ttnn.float32, FP32_TILE_BYTES, n_src, core_ranges),
        _cb(CB_COEF_COLS, ttnn.float32, FP32_TILE_BYTES, n_coef_cols * COEF_DEPTH, core_ranges),
        _cb(CB_F_TILES, input_tensor.dtype, f_tile, block_c * IO_DEPTH, core_ranges),
        _cb(CB_X_TILES, residual.dtype, x_tile, n * block_c * IO_DEPTH, core_ranges),
        _cb(CB_OUT_TILES, output_tensor.dtype, x_tile, n * block_c * OUT_DEPTH, core_ranges),
    ]

    # --- compile-time args ---
    reader_ct = [CB_COEF_SRC, CB_COEF_COLS, CB_F_TILES, CB_X_TILES, n, block_c, Ct, n_coef_cols, post_wt, comb_wt]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(residual).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(post).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(comb).get_compile_time_args())

    writer_ct = [CB_OUT_TILES, n, block_c, Ct]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [CB_COEF_COLS, CB_F_TILES, CB_X_TILES, CB_OUT_TILES, n, block_c, n_coef_cols]

    # --- runtime args (balanced contiguous 2-D split) ---
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for k in range(num_cores):
        x, y = k % gx, k // gx
        g_r, g_c = k // Q, k % Q
        row_start, row_end = (g_r * Rt) // R, ((g_r + 1) * Rt) // R
        c_start, c_end = (g_c * Ct) // Q, ((g_c + 1) * Ct) // Q
        num_rows, num_c = row_end - row_start, c_end - c_start
        reader_rt[x][y] = [
            input_tensor.buffer_address(),
            residual.buffer_address(),
            post.buffer_address(),
            comb.buffer_address(),
            row_start,
            num_rows,
            c_start,
            num_c,
            post_col,
            comb_col,
            g_r,
            g_c,
        ]
        writer_rt[x][y] = [output_tensor.buffer_address(), row_start, num_rows, c_start, num_c, g_r, g_c]
        compute_rt[x][y] = [num_rows, num_c]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # fp32 operands enter DEST exactly: every Float32 CB unpacked by the mix chain is tagged
    # UnpackToDestFp32 — CopyTile(F / X) and the UnaryBcast<Col> coefficient tiles. unary_bcast keys
    # its unpack-to-dest branch on the CB's unpack *dst* format, which is TF32 unless the CB is tagged
    # (measured: an untagged cb_coef_cols truncates every coefficient by ~2^-11).
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb_idx, dtype in (
        (CB_F_TILES, input_tensor.dtype),
        (CB_X_TILES, residual.dtype),
        (CB_COEF_COLS, ttnn.float32),
    ):
        if dtype == ttnn.float32:
            unpack_modes[cb_idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    kernel_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_config.math_fidelity,
        fp32_dest_acc_en=True,
        math_approx_mode=compute_config.math_approx_mode,
    )
    kernel_cfg.unpack_to_dest_mode = unpack_modes

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=kernel_cfg,
    )

    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
