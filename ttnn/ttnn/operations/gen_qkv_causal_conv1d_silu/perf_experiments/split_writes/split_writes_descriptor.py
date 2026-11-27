# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT split_writes — a copy of the op's create_program_descriptor with the q / k / v
tile writes shared between the writer (BRISC / NoC1) and the reader (NCRISC / NoC0).

Differences from the op's descriptor (gen_qkv_causal_conv1d_silu_program_descriptor.py), and nothing
else:
  * reader / writer kernel sources = kernels/sw_reader.cpp / kernels/sw_writer.cpp (this dir);
  * reader CT args += q / k / v TensorAccessorArgs + [q_tiles, k_tiles, v_tiles, out_group_tiles]
    (appended after tap3's accessor args, before the tap multicast attaches its own);
  * reader RT args += [q_addr, k_addr, v_addr] (appended after core_seq_tiles, before the multicast's);
  * one more CB: CB_WRITE_DONE (8), 16 B tokens, NCRISC -> writer, one per output group the NCRISC
    wrote into (depth = OUT_DEPTH output rows' worth of groups, capped);
  * defines SW_SHARE / SW_PERIOD / SW_POLL_BARRIER / SW_DYNAMIC on both DM kernels (static: the NCRISC
    owns SW_SHARE of every SW_PERIOD tiles, 0 = the baseline sequence through these kernels; dynamic:
    per group the NCRISC claims from the front and the writer from the back, + CB_CLAIM (9), 2 x 16 B).

make_create(share, period, poll_barrier) returns a drop-in for op_module.create_program_descriptor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu_program_descriptor as base

SW_KERNEL_DIR = Path(__file__).parent / "kernels"
CB_WRITE_DONE = 8
WRITE_DONE_PAGE_BYTES = 16
WRITE_DONE_MAX_DEPTH = 64
CB_CLAIM = 9  # dynamic mode: claim word A (reader) at +0, B (writer) at +16
CLAIM_PAGE_BYTES = 16


def gradient_share(lo: int, hi: int):
    """Per-core share rising from `lo` at the grid's top-left (logical (0, 0)) to `hi` at its
    bottom-right, linear in (x / (gx - 1) + y / (gy - 1)) / 2."""

    def share(x, y, gx, gy):
        t = ((x / (gx - 1) if gx > 1 else 0.0) + (y / (gy - 1) if gy > 1 else 0.0)) / 2
        return int(round(lo + (hi - lo) * t))

    return share


def make_create(share: int, period: int = 8, poll_barrier: bool = True, dynamic: bool = False, share_fn=None):
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

        cbs = [
            base._cb(index, page_bytes, pages, core_ranges, dtype)
            for index, page_bytes, pages in base.cb_layout(
                w, block_rows, plan.regime, split_rows=plan.split_rows, elem_bytes=elem_bytes, tile_bytes=tile_bytes
            )
        ]
        groups_per_row = base._div_up(w, plan.out_group_tiles)
        done_depth = min(WRITE_DONE_MAX_DEPTH, base.OUT_DEPTH * groups_per_row)
        cbs.append(base._cb(CB_WRITE_DONE, WRITE_DONE_PAGE_BYTES, done_depth, core_ranges, dtype))
        if dynamic:
            cbs.append(base._cb(CB_CLAIM, CLAIM_PAGE_BYTES, 2, core_ranges, dtype))

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
        ]
        for t in [input_tensor, history] + list(taps):
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        for t in outputs:  # split writes: the NCRISC's q / k / v accessors
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        reader_ct.extend([width_tiles[0], width_tiles[1], width_tiles[2], plan.out_group_tiles])

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
        ]
        for t in list(outputs) + [input_tensor]:
            writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        compute_ct = [w, block_rows, int(tile_input), plan.out_group_tiles, plan.silu_body, int(base.SILU_TILE_STALL)]

        reader_rt = ttnn.RuntimeArgs()
        writer_rt = ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        read_addrs = base.reader_addresses(input_tensor, history, taps)
        write_addrs = base.writer_addresses(input_tensor, outputs)
        out_addrs = [t.buffer_address() for t in outputs]
        input_col_offset_tiles = input_column_offset // TILE_DIM
        grid_rows = base._div_up(plan.num_cores, grid_x)
        for cw in work:
            core_share = [] if share_fn is None else [share_fn(cw.core.x, cw.core.y, grid_x, grid_rows)]
            ch_group_tile_start = cw.ch_group * w
            reader_rt[cw.core.x][cw.core.y] = (
                read_addrs
                + [ch_group_tile_start, input_col_offset_tiles, cw.core_seq_start, cw.core_seq_tiles]
                + out_addrs
                + core_share
            )
            writer_rt[cw.core.x][cw.core.y] = (
                write_addrs
                + [
                    ch_group_tile_start,
                    cw.core_seq_start,
                    cw.core_seq_tiles,
                    input_col_offset_tiles,
                ]
                + core_share
            )
            compute_rt[cw.core.x][cw.core.y] = [cw.core_seq_start, cw.core_seq_tiles]

        sw_defines = [
            ("SW_SHARE", str(int(share))),
            ("SW_PERIOD", str(int(period))),
            ("SW_POLL_BARRIER", str(int(poll_barrier))),
            ("SW_DYNAMIC", str(int(dynamic))),  # 0 static, 1 race (front / back), 2 overflow
            ("SW_SHARE_RT", str(int(share_fn is not None))),
        ]
        reader = ttnn.KernelDescriptor(
            kernel_source=str(SW_KERNEL_DIR / "sw_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=base._dm_config(ttnn.DataMovementProcessor.RISCV_1, base.READER_NOC),
        )
        reader.defines = list(sw_defines)
        writer = ttnn.KernelDescriptor(
            kernel_source=str(SW_KERNEL_DIR / "sw_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=base._dm_config(ttnn.DataMovementProcessor.RISCV_0, base.WRITER_NOC),
        )
        writer.defines = list(sw_defines)
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

    return create_program_descriptor
