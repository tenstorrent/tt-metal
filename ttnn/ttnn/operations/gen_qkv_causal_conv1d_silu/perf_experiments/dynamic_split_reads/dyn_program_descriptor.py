# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf experiment `dynamic_split_reads`: the op's create_program_descriptor with the dynamic split reader.

Identical to gen_qkv_causal_conv1d_silu_program_descriptor.create_program_descriptor except:
  - reader / writer kernels = kernels/dyn_reader.cpp / kernels/dyn_writer.cpp of this directory
    (the compute kernel is the op's own, unchanged);
  - plan.split_rows > 0 switches the dynamic split ON (its value is not used as a ratio any more);
  - cb_split_requests is one 32 B page (the reader's shared words R, D) and cb_split_done one 16 B
    page (the writer's shared word W) — pushed once at kernel start, never popped;
  - new CT args (module knobs below): reader [9] READ_WINDOW, [10] CLAIM_CHUNK, [11] HEAD_STICKS, [12] TAIL_WINDOW, [13] DUP_GRACE,
    [14] ROWS_DEPTH (slot-reuse guard ring);
    writer [10] WRITER_WINDOW, [11] CLAIM_CHUNK, [12] MIN_ENTER, [13] POLL_IN_FLUSH, [14] GATE_WRITES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu_program_descriptor as base

KERNEL_DIR = Path(__file__).parent / "kernels"

# Knobs of the dynamic split (bench variants override them through monkeypatch). Defaults = the variant
# measured best on conv1d-004 (dyn_r32_w8_c2_g): an eager reader, a writer that only helps while idle and
# counts its unacked output writes against an 8-stick window.
READ_WINDOW = 32  # reader: max NoC0 reads outstanding when it claims the next chunk of the head
HEAD_STICKS = 32  # reader: sticks [0, HEAD_STICKS) of a tile-row are claimed under READ_WINDOW ...
TAIL_WINDOW = 8  # ... the rest only while at most TAIL_WINDOW of its reads are outstanding
DUP_GRACE = 0xFFFFFFFF  # reader polls to wait for the writer's share before re-reading it (0xFFFFFFFF: never)
WRITER_WINDOW = 8  # writer: max NoC1 reads (+ unacked writes if GATE_WRITES) outstanding when it claims
CLAIM_CHUNK = 2  # sticks per claim (divides 32)
MIN_ENTER = 4  # writer enters an OPEN tile-row only if at least this many sticks are unclaimed
POLL_IN_FLUSH = False  # writer also serves claims while its output writes drain
GATE_WRITES = True  # writer's window also counts its unacked output writes (weighted tile/stick bytes)

SPLIT_STATE_PAGE_BYTES = {base.CB_SPLIT_REQUESTS: 32, base.CB_SPLIT_DONE: 16}


def create_program_descriptor(
    input_tensor,
    history,
    taps,
    outputs,
    widths,
    plan,
    *,
    grid_x: int,
    math_fidelity,
    fp32_dest_acc_en: bool,
    dst_full_sync_en: bool,
    input_column_offset: int = 0,
    page_channels: Optional[int] = None,
):
    TILE_DIM = base.TILE_DIM
    dtype = input_tensor.dtype
    elem_bytes = input_tensor.element_size()
    tile_bytes = ttnn.tile_size(dtype)
    tensor_seq_tiles = int(input_tensor.shape[1]) // TILE_DIM
    input_width_tiles = base._div_up(int(input_tensor.shape[2]), TILE_DIM)
    width_tiles = [x // TILE_DIM for x in widths]
    w = plan.block_ch_tiles
    block_rows = plan.block_rows
    tile_input = plan.regime == base.REGIME_TILE

    work = base.assign_work(plan, tensor_seq_tiles, grid_x)
    core_ranges = base._row_wise_core_range_set(plan.num_cores, grid_x)

    cbs = []
    for index, page_bytes, pages in base.cb_layout(
        w, block_rows, plan.regime, split_rows=plan.split_rows, elem_bytes=elem_bytes, tile_bytes=tile_bytes
    ):
        if index in SPLIT_STATE_PAGE_BYTES:
            page_bytes, pages = SPLIT_STATE_PAGE_BYTES[index], 1
        cbs.append(base._cb(index, page_bytes, pages, core_ranges, dtype))

    history_page_channels = 0 if page_channels is None else int(page_channels)
    reader_ct = [
        w,
        block_rows,
        int(tile_input),
        input_width_tiles,
        elem_bytes,
        tile_bytes,
        history_page_channels,
        plan.split_rows,
        int(base.STAGGER_ROW_WALK),
        READ_WINDOW,
        CLAIM_CHUNK,
        HEAD_STICKS,
        TAIL_WINDOW,
        DUP_GRACE,
        base.ROWS_DEPTH,
    ]
    for t in [input_tensor, history] + list(taps):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    stick_bytes = w * TILE_DIM * elem_bytes
    writer_ct = [
        w,
        width_tiles[0],
        width_tiles[1],
        width_tiles[2],
        tile_bytes,
        block_rows,
        plan.split_rows,
        stick_bytes,
        plan.out_group_tiles,
        int(base.STAGGER_ROW_WALK),
        WRITER_WINDOW,
        CLAIM_CHUNK,
        MIN_ENTER,
        int(POLL_IN_FLUSH),
        int(GATE_WRITES),
    ]
    for t in list(outputs) + [input_tensor]:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [w, block_rows, int(tile_input), plan.out_group_tiles, plan.silu_body, int(base.SILU_TILE_STALL)]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    read_addrs = base.reader_addresses(input_tensor, history, taps)
    write_addrs = base.writer_addresses(input_tensor, outputs)
    input_col_offset_tiles = input_column_offset // TILE_DIM
    for cw in work:
        ch_group_tile_start = cw.ch_group * w
        reader_rt[cw.core.x][cw.core.y] = read_addrs + [
            ch_group_tile_start,
            input_col_offset_tiles,
            cw.core_seq_start,
            cw.core_seq_tiles,
        ]
        writer_rt[cw.core.x][cw.core.y] = write_addrs + [
            ch_group_tile_start,
            cw.core_seq_start,
            cw.core_seq_tiles,
            input_col_offset_tiles,
        ]
        compute_rt[cw.core.x][cw.core.y] = [cw.core_seq_start, cw.core_seq_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "dyn_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=base._dm_config(ttnn.DataMovementProcessor.RISCV_1, base.READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "dyn_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=base._dm_config(ttnn.DataMovementProcessor.RISCV_0, base.WRITER_NOC),
    )
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_fidelity = math_fidelity
    compute_config.fp32_dest_acc_en = bool(fp32_dest_acc_en)
    compute_config.dst_full_sync_en = bool(dst_full_sync_en)
    compute_config.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(base.KERNEL_DIR / "gen_qkv_causal_conv1d_silu_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    taps_mcast = ttnn.Mcast(
        input_tensor.device(),
        ttnn.McastConfig(noc=base.READER_NOC),
        core_ranges,
        plan.cores_per_group,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.RowMajor,
    )
    descriptor = ttnn.ProgramDescriptor(cbs=cbs)
    taps_mcast.attach(descriptor, base.TAPS_MCAST_PREFIX, [reader], 0)
    kernels = [None, None, compute]
    kernels[base.READER_KERNEL_INDEX] = reader
    kernels[base.WRITER_KERNEL_INDEX] = writer
    descriptor.kernels = kernels
    return descriptor, work
