# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_nhwc — blocking selection + ProgramDescriptor (regime R0).

Every knob (core split, bm / bk / bn, buffer depths) is defined once here and
every CB size, loop trip count and kernel arg derives from it.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
L1_NOC_ALIGN = 16

# ---- Module knobs (single source of truth) ----
CB_BUDGET_BYTES = 768 * 1024  # preferred per-core CB budget (perf lamp L5)
L1_HEADROOM_BYTES = 128 * 1024  # kept free under the hard cap (L1_SMALL, allocator slack)


def l1_cap_bytes():
    """Hard per-core CB cap: worker L1 unreserved size minus headroom."""
    return ttnn.get_max_worker_l1_unreserved_size() - L1_HEADROOM_BYTES


ACT_RM_DEPTH = 2  # cb_act_rm depth, in (1 tile-row x bk) quanta
WEIGHT_DEPTH = 2  # cb_weight_tiles depth, in (bk x bn) K-blocks
OUT_RM_DEPTH = 2  # cb_out_rm depth, in (1 tile-row x bn) quanta
BK_FLOOR = 4  # bk >= min(K_t, BK_FLOOR) on the first walk (perf lamp L7)
# Compute config: one source for the descriptor, the DEST capacity used by the
# subblock rule and the accumulator page format.
MATH_FIDELITY = ttnn.MathFidelity.HiFi4
FP32_DEST_ACC_EN = True
DST_TILES = 4 if FP32_DEST_ACC_EN else 8  # half-sync DEST capacity in tiles
ACC_DTYPE = ttnn.float32 if FP32_DEST_ACC_EN else ttnn.bfloat16

# ---- CB indices (semantic names) ----
CB_ACT_WINDOW = 0
CB_ACT_RM = 1
CB_WEIGHT_TILES = 2
CB_BIAS_TILES = 3
CB_ACT_TILES = 4
CB_PARTIALS = 5
CB_BIASED = 6
CB_OUT_RM = 7


def _div_up(a, b):
    return -(-a // b)


def tile_bytes(dtype):
    """Bytes of one 32x32 tile of `dtype` (bf8b: 64 B exponents + 1024 B mantissas)."""
    return {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}[dtype]


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def conv_geometry(*, C_in, kH, kW, groups, elem_bytes):
    """Prepared-K geometry shared by prepare_conv2d_weights and the kernels."""
    granule = L1_NOC_ALIGN // elem_bytes
    C_in_g = C_in // groups
    tap_width = _div_up(C_in_g, granule) * granule
    band_tiles = _div_up(kW * tap_width, TILE)
    return {"tap_width": tap_width, "band_tiles": band_tiles, "K_t": kH * band_tiles}


def _window_bounds(pix0, pix1, *, H, W, H_out, W_out, kH, kW, padding, stride, dilation):
    """Input window an m-block's pixels [pix0, pix1] read (regime R0 + R10 column clip).

    Returns (row_lo, row_hi, col_lo, last_hi, win_w, mid_lo, mid_hi) over flattened input
    rows (n*H + h). Window stick of input (row, col) = (row - row_lo) * win_w + (col - col_lo).
      * one output row  -> 2-D clip: every row holds columns [col_lo, last_hi], win_w = width.
      * several rows    -> contiguous flattened range, win_w = W: the first input row starts
        at col_lo and the last ends at last_hi (each only when that row is not padding-clamped,
        i.e. read by that one output row alone); middle rows are whole.
    Mirrors the reader kernel's window bound computation exactly."""
    hw_out = H_out * W_out
    n0, r0 = divmod(pix0, hw_out)
    n1, r1 = divmod(pix1, hw_out)
    ho0, wo0 = divmod(r0, W_out)
    ho1, wo1 = divmod(r1, W_out)
    raw_lo = ho0 * stride - padding
    raw_hi = ho1 * stride - padding + dilation * (kH - 1)
    row_lo = n0 * H + min(max(0, raw_lo), H - 1)
    row_hi = max(n1 * H + max(min(H - 1, raw_hi), 0), row_lo)
    c_first = min(max(0, wo0 * stride - padding), W - 1)
    c_last = max(min(W - 1, wo1 * stride - padding + dilation * (kW - 1)), 0)
    if n0 == n1 and ho0 == ho1:
        col_lo, last_hi = c_first, max(c_last, c_first)
        return row_lo, row_hi, col_lo, last_hi, last_hi - col_lo + 1, col_lo, last_hi
    col_lo = c_first if raw_lo >= 0 else 0
    last_hi = c_last if raw_hi <= H - 1 else W - 1
    return row_lo, row_hi, col_lo, last_hi, W, 0, W - 1


def _window_sticks(pix0, pix1, **kw):
    row_lo, row_hi, col_lo, last_hi, win_w, _, _ = _window_bounds(pix0, pix1, **kw)
    return (row_hi - row_lo) * win_w + (last_hi - col_lo) + 1


def select_blocking(
    *,
    N,
    H,
    W,
    C_in,
    C_out,
    kH,
    kW,
    padding,
    stride,
    dilation,
    groups,
    elem_bytes,
    weight_tile_bytes,
    num_cores,
    has_bias,
    l1_cap_bytes,
):
    geo = conv_geometry(C_in=C_in, kH=kH, kW=kW, groups=groups, elem_bytes=elem_bytes)
    tap_width, band_tiles, K_t = geo["tap_width"], geo["band_tiles"], geo["K_t"]
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    M = N * H_out * W_out
    M_t = _div_up(M, TILE)
    N_t = _div_up(C_out, TILE)

    # Core split: m first, n only over the cores the m split leaves idle.
    core_m_tiles = _div_up(M_t, min(M_t, num_cores))
    g_m = _div_up(M_t, core_m_tiles)
    n_budget = max(1, num_cores // g_m)
    core_n_tiles = next(d for d in _divisors(N_t) if N_t // d <= n_budget)
    g_n = N_t // core_n_tiles

    T = TILE * TILE * elem_bytes  # act / out tile bytes
    T_w = weight_tile_bytes  # weight / bias tile bytes (weight_dtype)
    T_acc = tile_bytes(ACC_DTYPE)  # accumulator page = DEST width
    win_kw = dict(H=H, W=W, H_out=H_out, W_out=W_out, kH=kH, kW=kW, padding=padding, stride=stride, dilation=dilation)

    def window_max(bm):
        best = 1
        for mi in range(g_m):
            m_start = mi * core_m_tiles
            m_tiles = min(core_m_tiles, M_t - m_start)
            for b in range(_div_up(m_tiles, bm)):
                t0 = m_start + b * bm
                t1 = min(t0 + bm, m_start + m_tiles)
                best = max(best, _window_sticks(t0 * TILE, min(t1 * TILE, M) - 1, **win_kw))
        return best

    def footprint(bm, bk, bn, win):
        return (
            win * tap_width * elem_bytes
            + ACT_RM_DEPTH * bk * T
            + bm * bk * T
            + WEIGHT_DEPTH * bk * bn * T_w
            + bm * bn * T_acc
            + (core_n_tiles * T_w if has_bias else 0)
            + OUT_RM_DEPTH * bn * T
        )

    k_divs_desc = sorted(_divisors(K_t), reverse=True)
    win_cache = {}

    def win_of(bm):
        if bm not in win_cache:
            win_cache[bm] = window_max(bm)
        return win_cache[bm]

    def walk(budget):
        for bn in sorted(_divisors(core_n_tiles), reverse=True):
            # Step 2 (bn = core_n_tiles only): bm = core_m_tiles with the bk floor.
            if bn == core_n_tiles:
                bm = core_m_tiles
                win = win_of(bm)
                for bk in k_divs_desc:
                    if bk >= min(K_t, BK_FLOOR) and footprint(bm, bk, bn, win) <= budget:
                        return bm, bk, bn, win
            # Step 3: walk bm down, bk over all divisors.
            for bm in range(core_m_tiles, 0, -1):
                win = win_of(bm)
                bk = next((d for d in k_divs_desc if footprint(bm, d, bn, win) <= budget), None)
                if bk is not None:
                    return bm, bk, bn, win
        return None

    # The preferred budget first (perf lamp L5: footprint is a target); only a shape that
    # cannot block under it falls back to the hard per-core L1 cap.
    choice = walk(CB_BUDGET_BYTES) or walk(l1_cap_bytes)
    if choice is None:
        raise NotImplementedError("conv2d_nhwc: input window exceeds L1 — DRAM-slicing regime R10 not built")
    bm, bk, bn, window_sticks_max = choice

    sb_w = max(d for d in _divisors(bn) if d <= DST_TILES)
    sb_h = max(d for d in _divisors(bm) if d * sb_w <= DST_TILES)

    return {
        **geo,
        "H_out": H_out,
        "W_out": W_out,
        "M": M,
        "M_t": M_t,
        "N_t": N_t,
        "N": N,
        "H": H,
        "W": W,
        "C_in": C_in,
        "C_out": C_out,
        "kH": kH,
        "kW": kW,
        "padding": padding,
        "stride": stride,
        "dilation": dilation,
        "elem_bytes": elem_bytes,
        "has_bias": has_bias,
        "core_m_tiles": core_m_tiles,
        "g_m": g_m,
        "core_n_tiles": core_n_tiles,
        "g_n": g_n,
        "block_m_tiles": bm,
        "block_k_tiles": bk,
        "block_n_tiles": bn,
        "num_k_blocks": K_t // bk,
        "num_n_blocks": core_n_tiles // bn,
        "window_sticks_max": window_sticks_max,
        "out_subblock_h": sb_h,
        "out_subblock_w": sb_w,
        "T": T,
        "T_w": T_w,
        "T_acc": T_acc,
        "footprint": footprint(bm, bk, bn, window_sticks_max),
    }


def _cb(index, dtype, page_size, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(input_tensor, weight_tensor, bias_tensor, output_tensor, plan, grid):
    p = plan
    bm, bk, bn = p["block_m_tiles"], p["block_k_tiles"], p["block_n_tiles"]
    has_bias = p["has_bias"]
    act_dtype = input_tensor.dtype
    w_dtype = weight_tensor.dtype
    eb = p["elem_bytes"]

    # ---- Core list: first g_m * g_n cores of the grid, row-major ----
    num_active = p["g_m"] * p["g_n"]
    coords = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(num_active)]
    ranges = []
    full_rows, rem = divmod(num_active, grid.x)
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    cores = ttnn.CoreRangeSet(ranges)

    # ---- Circular buffers (sizes derive from the knobs) ----
    tap_bytes = p["tap_width"] * eb
    cbs = [
        _cb(CB_ACT_WINDOW, act_dtype, tap_bytes, p["window_sticks_max"], cores),
        _cb(CB_ACT_RM, act_dtype, p["T"], ACT_RM_DEPTH * bk, cores),
        _cb(CB_WEIGHT_TILES, w_dtype, p["T_w"], WEIGHT_DEPTH * bk * bn, cores),
        _cb(CB_ACT_TILES, act_dtype, p["T"], bm * bk, cores),
        _cb(CB_OUT_RM, output_tensor.dtype, p["T"], OUT_RM_DEPTH * bn, cores),
    ]
    partial_formats = [ttnn.CBFormatDescriptor(buffer_index=CB_PARTIALS, data_format=ACC_DTYPE, page_size=p["T_acc"])]
    if has_bias:
        cbs.append(_cb(CB_BIAS_TILES, bias_tensor.dtype, p["T_w"], p["core_n_tiles"], cores))
        # cb_biased aliases cb_partials: one allocation, two FIFOs (in-place bias add).
        partial_formats.append(
            ttnn.CBFormatDescriptor(buffer_index=CB_BIASED, data_format=ACC_DTYPE, page_size=p["T_acc"])
        )
    cbs.append(
        ttnn.CBDescriptor(total_size=bm * bn * p["T_acc"], core_ranges=cores, format_descriptors=partial_formats)
    )

    # ---- Reader (NCRISC): window fetch + im2col gather ----
    reader_ct = [
        CB_ACT_WINDOW,
        CB_ACT_RM,
        bm,
        bk,
        p["num_k_blocks"],
        p["num_n_blocks"],
        p["band_tiles"],
        p["tap_width"],
        eb,
        p["C_in"] * eb,  # bytes of one DRAM input stick
        p["kH"],
        p["kW"],
        p["H"],
        p["W"],
        p["H_out"],
        p["W_out"],
        p["M"],
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---- Writer (BRISC): weights, bias, output ----
    writer_ct = [
        CB_WEIGHT_TILES,
        CB_BIAS_TILES,
        CB_OUT_RM,
        bm,
        bk,
        bn,
        p["num_k_blocks"],
        p["num_n_blocks"],
        p["N_t"],
        p["core_n_tiles"],
        1 if has_bias else 0,
        p["M"],
        eb,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(bias_tensor).get_compile_time_args()
        if has_bias
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- Compute ----
    compute_ct = [
        CB_ACT_RM,
        CB_WEIGHT_TILES,
        CB_BIAS_TILES,
        CB_ACT_TILES,
        CB_PARTIALS,
        CB_BIASED,
        CB_OUT_RM,
        bm,
        bk,
        bn,
        p["num_k_blocks"],
        p["num_n_blocks"],
        p["out_subblock_h"],
        p["out_subblock_w"],
        1 if has_bias else 0,
        p["core_n_tiles"],
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    w_addr = weight_tensor.buffer_address()
    b_addr = bias_tensor.buffer_address() if has_bias else 0
    out_addr = output_tensor.buffer_address()
    for c, xy in enumerate(coords):
        mi, ni = divmod(c, p["g_n"])
        m_tile_start = mi * p["core_m_tiles"]
        m_tiles = min(p["core_m_tiles"], p["M_t"] - m_tile_start)
        num_m_blocks = _div_up(m_tiles, bm)
        n_tile_start = ni * p["core_n_tiles"]
        reader_rt[xy.x][xy.y] = [
            in_addr,
            m_tile_start,
            m_tiles,
            num_m_blocks,
            p["padding"],
            p["stride"],
            p["dilation"],
        ]
        writer_rt[xy.x][xy.y] = [w_addr, b_addr, out_addr, m_tile_start, m_tiles, num_m_blocks, n_tile_start]
        compute_rt[xy.x][xy.y] = [num_m_blocks]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_nhwc_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=MATH_FIDELITY, fp32_dest_acc_en=FP32_DEST_ACC_EN),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
