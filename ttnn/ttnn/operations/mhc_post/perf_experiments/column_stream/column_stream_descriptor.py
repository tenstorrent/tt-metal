# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post COLUMN-STREAM candidate ProgramDescriptor (perf_experiments/column_stream).

Same work split, coefficient path (writer-side COEF_EXPANDER) and block-size policy as the op. What changes:
the reader -> compute -> writer handoff unit is a GROUP of G columns (G <= B), and the reader keeps up to
INFLIGHT_COLS columns of reads in flight (trid-tracked, pushed in order as each group lands), independent of G.
CB capacities are expressed in columns (IN_CAP_COLS / OUT_CAP_COLS, default DEPTH * B = the op's footprint),
rounded up to a multiple of G so every G-aligned push is wrap-exact.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

from . import baseline_descriptor as base

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- candidate knobs (None = derive from the op's block B) ----
KNOBS = {
    "G": 1,  # columns per CB handoff
    "INFLIGHT_COLS": None,  # columns of reads in flight (default B); capped at 16 groups (trids) and IN_CAP_COLS
    "IN_CAP_COLS": None,  # input CB capacity in columns (default DEPTH_IN * B)
    "OUT_CAP_COLS": None,  # output CB capacity in columns (default DEPTH_OUT * B)
    "WRITER_FLUSH": True,  # per-window noc_async_writes_flushed (True) or write barrier (False) before pop
    "WG": None,  # writer window in columns (multiple of G; default G)
    "TAIL_COLS": 0,  # the core's last TAIL_COLS columns: reader batches and writer windows shrink to one group
    "RB": None,  # reader issue batch in columns (multiple of G; default G): term-major request order over the batch
    "STUB_DM": False,  # ablation: reader/writer skip the data reads/writes (CB protocol kept)
    "STUB_COMPUTE": False,  # ablation: compute skips the mix (CB protocol kept)
}
NUM_TRIDS = 15  # trids 1..15 (0 = untagged default)


def _round_up(x, m):
    return (x + m - 1) // m * m


def op_block_col_tiles(n, col_tiles_per_row, assignment, pages):
    sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles = pages
    fit = base._block_col_tiles_fit(n, sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles)
    B = min(fit, base._max_segment_col_tiles(assignment, col_tiles_per_row))
    B = min(B, math.ceil(max(c for _, _, c in assignment) / base.MIN_BLOCKS_PER_CORE))
    if base.MAX_BLOCK_COL_TILES is not None:
        B = min(B, base.MAX_BLOCK_COL_TILES)
    return max(1, B)


def create_program_descriptor(input_tensor, residual, post, comb, output_tensor, compute_kernel_config, knobs=None):
    k = dict(KNOBS)
    k.update(knobs or {})
    device = input_tensor.device()
    TILE_HW = base.TILE_HW

    n = post.shape[-1]
    col_tiles_per_row = input_tensor.shape[-1] // TILE_HW
    token_tiles = base._tensor_token_tiles(list(input_tensor.padded_shape))
    total_units = token_tiles * col_tiles_per_row
    post_tiles_per_row = math.ceil(n / TILE_HW)
    comb_tiles_per_row = math.ceil(n * n / TILE_HW)
    num_raw_tiles = post_tiles_per_row + comb_tiles_per_row
    coef_tiles_per_stream = math.ceil((n + 1) / 2)
    num_coef_tiles = n * coef_tiles_per_stream

    sublayer_page = input_tensor.buffer_page_size()
    residual_page = residual.buffer_page_size()
    output_page = output_tensor.buffer_page_size()
    coef_page = post.buffer_page_size()

    grid_size = device.compute_with_storage_grid_size()
    all_cores, assignment = base._work_assignment(grid_size, total_units)
    B = op_block_col_tiles(
        n, col_tiles_per_row, assignment, (sublayer_page, residual_page, coef_page, num_coef_tiles, num_raw_tiles)
    )
    G = max(1, min(k["G"], B))
    in_cap = _round_up(k["IN_CAP_COLS"] or base.DEPTH_IN * B, G)
    out_cap = _round_up(k["OUT_CAP_COLS"] or base.DEPTH_OUT * B, G)

    def cols(v, default):  # knob value: None -> default, "B" -> the op's block
        return default if v is None else (B if v == "B" else v)

    WG = _round_up(min(cols(k["WG"], G), out_cap), G)
    inflight_cols = k["INFLIGHT_COLS"] or B
    inflight_groups = max(1, min(NUM_TRIDS, inflight_cols // G, in_cap // G))
    RB = _round_up(min(cols(k["RB"], G), inflight_groups * G), G)
    create_program_descriptor.last = dict(
        B=B, G=G, in_cap=in_cap, out_cap=out_cap, WG=WG, RB=RB, inflight_groups=inflight_groups
    )

    cbs = [
        base._cb(base.CB_SUBLAYER_TILES, input_tensor.dtype, sublayer_page, in_cap, all_cores),
        base._cb(base.CB_RESIDUAL_TILES, residual.dtype, residual_page, n * in_cap, all_cores),
        base._cb(base.CB_COEF_RAW, post.dtype, coef_page, num_raw_tiles, all_cores),
        base._cb(base.CB_COEF_BCAST, post.dtype, coef_page, base.COEF_DEPTH * num_coef_tiles, all_cores),
        base._cb(base.CB_OUTPUT_TILES, output_tensor.dtype, output_page, n * out_cap, all_cores),
    ]

    reader_ct = [
        n,
        col_tiles_per_row,
        G,
        sublayer_page,
        residual_page,
        base.CB_SUBLAYER_TILES,
        base.CB_RESIDUAL_TILES,
        inflight_groups,
        RB,
        k["TAIL_COLS"],
    ]
    for t in (input_tensor, residual):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    writer_ct = [
        n,
        col_tiles_per_row,
        G,
        output_page,
        base.CB_OUTPUT_TILES,
        post_tiles_per_row,
        comb_tiles_per_row,
        coef_page,
        TILE_HW,
        base.CB_COEF_RAW,
        base.CB_COEF_BCAST,
        coef_tiles_per_stream,
        1,  # COEF_EXPANDER == writer (the only mode this candidate implements)
        int(bool(k["WRITER_FLUSH"])),
        WG,
        k["TAIL_COLS"],
    ]
    for t in (output_tensor, post, comb):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    compute_ct = [
        n,
        col_tiles_per_row,
        G,
        base.CB_SUBLAYER_TILES,
        base.CB_RESIDUAL_TILES,
        base.CB_COEF_BCAST,
        base.CB_OUTPUT_TILES,
        coef_tiles_per_stream,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    f_addr, x_addr = input_tensor.buffer_address(), residual.buffer_address()
    p_addr, m_addr = post.buffer_address(), comb.buffer_address()
    o_addr = output_tensor.buffer_address()
    for core, start, count in assignment:
        reader_rt[core.x][core.y] = [f_addr, x_addr, start, count]
        writer_rt[core.x][core.y] = [o_addr, start, count, p_addr, m_addr]
        compute_rt[core.x][core.y] = [start, count]

    unpack_modes = [ttnn.UnpackToDestMode.Default] * base.NUM_CIRCULAR_BUFFERS
    for index, dtype in (
        (base.CB_SUBLAYER_TILES, input_tensor.dtype),
        (base.CB_RESIDUAL_TILES, residual.dtype),
        (base.CB_COEF_BCAST, post.dtype),
    ):
        if dtype == ttnn.float32:
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
    )
    compute_cfg.dst_full_sync_en = base.DST_FULL_SYNC
    compute_cfg.unpack_to_dest_mode = unpack_modes

    dm_defines = [("CS_STUB_DM", "1")] if k["STUB_DM"] else []
    reader_defines = dm_defines + [(d, "1") for d in k.get("READER_DEFINES", ())]
    cmp_defines = [("CS_STUB_COMPUTE", "1")] if k["STUB_COMPUTE"] else []
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mhc_post_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=reader_defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mhc_post_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=dm_defines,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mhc_post_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=cmp_defines,
            config=compute_cfg,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
