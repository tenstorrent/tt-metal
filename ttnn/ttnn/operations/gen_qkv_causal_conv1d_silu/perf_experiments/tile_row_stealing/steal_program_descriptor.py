# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf experiment `tile_row_stealing`: the op's create_program_descriptor with cross-core tile-row stealing.

Identical to gen_qkv_causal_conv1d_silu_program_descriptor.create_program_descriptor except:
  - reader / writer / compute kernels = kernels/steal_{reader,writer,compute}.cpp of this directory;
  - STEAL_MODE (module knob, bench variants override it):
        None  auto: STEAL_ON when some core owns more than one block (blocks_max > 1) and a group has
              more than one core, else STEAL_STATIC — a one-block-per-core plan has nothing to steal by
              construction (every core claims its only block at kernel start, before any core can have
              finished one), so it runs the op's static path with zero protocol cost
        0     STEAL_STATIC: the op's static token ranges (baseline code path in all three kernels)
        1     STEAL_OWN: dynamic block schedule + atomic claims of the own range only (protocol overhead)
        2     STEAL_ON: own range front-first, then steal from the back of the group's other ranges
  - dynamic modes add: a claim semaphore (id 0, initial 0; the tap multicast's semaphores start at 1),
    cb_ctrl (CB 8, reader -> compute block messages), cb_steal_scratch (CB 9, atomic return + probe words),
    and cb_split_requests always (the writer learns each tile-row's index from it);
  - reader CT args [9] steal_mode, [10] STEAL_MIN, [11] claim semaphore id (accessors from 12);
    writer CT [10] dynamic (accessors from 11); compute CT [6] dynamic;
  - reader RT args gain the group geometry: group_first_core, cores_per_group, own slot, range base / rem
    (slot s owns base + (s < rem) tile-rows), grid_x, grid rows used, then the virtual NoC x of every
    logical column and y of every logical row (worker_core_from_logical_core; checked separable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu_program_descriptor as base

KERNEL_DIR = Path(__file__).parent / "kernels"

STEAL_STATIC = 0
STEAL_OWN = 1
STEAL_ON = 2

STEAL_MODE: Optional[int] = None  # None: auto (see module doc)
STEAL_MIN = 1  # steal only from a range with at least this many unclaimed blocks

CB_CTRL = 8
CB_STEAL_SCRATCH = 9
CTRL_PAGE_BYTES = 16
CLAIM_SEM_ID = 0
SCRATCH_STRIDE = 16


def resolve_steal_mode(plan) -> int:
    if STEAL_MODE is not None:
        return STEAL_MODE
    blocks_max = base._div_up(plan.core_seq_tiles_max, plan.block_rows)
    return STEAL_ON if (blocks_max > 1 and plan.cores_per_group > 1) else STEAL_STATIC


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
    steal_mode = resolve_steal_mode(plan)
    dynamic = steal_mode != STEAL_STATIC
    device = input_tensor.device()

    work = base.assign_work(plan, tensor_seq_tiles, grid_x)
    core_ranges = base._row_wise_core_range_set(plan.num_cores, grid_x)

    # ---- circular buffers ----
    cbs = []
    have_requests = False
    for index, page_bytes, pages in base.cb_layout(
        w, block_rows, plan.regime, split_rows=plan.split_rows, elem_bytes=elem_bytes, tile_bytes=tile_bytes
    ):
        if index == base.CB_SPLIT_REQUESTS:
            have_requests = True
            if dynamic:
                pages = max(pages, 2 * block_rows + 2)
        cbs.append(base._cb(index, page_bytes, pages, core_ranges, dtype))
    if dynamic:
        if not have_requests:
            cbs.append(
                base._cb(
                    base.CB_SPLIT_REQUESTS, base.SPLIT_HANDSHAKE_PAGE_BYTES, 2 * block_rows + 2, core_ranges, dtype
                )
            )
        cbs.append(base._cb(CB_CTRL, CTRL_PAGE_BYTES, base.ROWS_DEPTH + 2, core_ranges, dtype))
        cbs.append(base._cb(CB_STEAL_SCRATCH, SCRATCH_STRIDE, plan.cores_per_group + 1, core_ranges, dtype))

    # ---- reader ----
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
        steal_mode,
        STEAL_MIN,
        CLAIM_SEM_ID,
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
        int(dynamic),
    ]
    for t in list(outputs) + [input_tensor]:
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [
        w,
        block_rows,
        int(tile_input),
        plan.out_group_tiles,
        plan.silu_body,
        int(base.SILU_TILE_STALL),
        int(dynamic),
    ]

    # ---- group geometry for the claim protocol (virtual NoC coordinates, separable per axis) ----
    grid_rows = base._div_up(plan.num_cores, grid_x)
    xs = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(grid_x)]
    ys = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(grid_rows)]
    if dynamic:
        for cw in work:
            v = device.worker_core_from_logical_core(cw.core)
            assert (int(v.x), int(v.y)) == (xs[cw.core.x], ys[cw.core.y]), "virtual coordinates not separable"
    range_base, range_rem = divmod(tensor_seq_tiles, plan.cores_per_group)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    read_addrs = base.reader_addresses(input_tensor, history, taps)
    write_addrs = base.writer_addresses(input_tensor, outputs)
    input_col_offset_tiles = input_column_offset // TILE_DIM
    for cw in work:
        ch_group_tile_start = cw.ch_group * w
        slot = (cw.core.y * grid_x + cw.core.x) - cw.ch_group * plan.cores_per_group
        reader_rt[cw.core.x][cw.core.y] = (
            read_addrs
            + [ch_group_tile_start, input_col_offset_tiles, cw.core_seq_start, cw.core_seq_tiles]
            + [
                cw.ch_group * plan.cores_per_group,
                plan.cores_per_group,
                slot,
                range_base,
                range_rem,
                grid_x,
                grid_rows,
            ]
            + xs
            + ys
        )
        writer_rt[cw.core.x][cw.core.y] = write_addrs + [
            ch_group_tile_start,
            cw.core_seq_start,
            cw.core_seq_tiles,
            input_col_offset_tiles,
        ]
        compute_rt[cw.core.x][cw.core.y] = [cw.core_seq_start, cw.core_seq_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "steal_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=base._dm_config(ttnn.DataMovementProcessor.RISCV_1, base.READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "steal_writer.cpp"),
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
        kernel_source=str(KERNEL_DIR / "steal_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    taps_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=base.READER_NOC),
        core_ranges,
        plan.cores_per_group,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.RowMajor,
    )
    descriptor = ttnn.ProgramDescriptor(cbs=cbs)
    if dynamic:
        descriptor.semaphores = [ttnn.SemaphoreDescriptor(id=CLAIM_SEM_ID, core_ranges=core_ranges, initial_value=0)]
    taps_mcast.attach(descriptor, base.TAPS_MCAST_PREFIX, [reader], CLAIM_SEM_ID + 1 if dynamic else 0)
    kernels = [None, None, compute]
    kernels[base.READER_KERNEL_INDEX] = reader
    kernels[base.WRITER_KERNEL_INDEX] = writer
    descriptor.kernels = kernels
    return descriptor, work
