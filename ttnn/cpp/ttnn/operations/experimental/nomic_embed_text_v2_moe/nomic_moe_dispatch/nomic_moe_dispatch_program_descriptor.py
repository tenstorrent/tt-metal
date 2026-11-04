# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_moe_dispatch ProgramDescriptor (regime R1 `local_tilize_two_noc`, op_design.md).

Kernels (all on the 11x10 grid; per-core work is derived on device from the logical core and the CT args):
  * nomic_moe_dispatch_reader.cpp      RISCV_0 / NOC0: rows [0, rows_noc0) of every block, into cb_rows_head
  * nomic_moe_dispatch_reader_noc1.cpp RISCV_1 / NOC1: rows [rows_noc0, 32) of every block, into cb_rows_tail
  * nomic_moe_dispatch_compute.cpp     fast tilize of each 32 x w block into cb_output_tiles (the output shard)
There is no writer kernel: the output is produced in its home L1 (the RISC a writer would use reads on NOC1).
gather_index (Refinement 3, R10 `ids_row_mcast`): each grid row's x = 0 core (its NOC0 reader) reads the row's span of
ids a round at a time and multicasts it along the row into cb_gather_ids, behind semaphore IDS_READY_SEMAPHORE_ID.

The per-shape parts (CB layouts, CT args) are built once per plan and reused; a call only supplies the two
buffer addresses (common RT args) and the output tensor that backs cb_output_tiles.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import ttnn

from .nomic_moe_dispatch import (
    GRID_X,
    GRID_Y,
    ID_BYTES,
    IDS_ALIGN,
    NUM_CORES,
    ROW_BYTES,
    TILE_BYTES,
    Plan,
    _grid,
    _ids_cb_bytes,
    _rows_cb_bytes,
    default_compute_kernel_config,
)

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; op_design.md Circular Buffers)
CB_ROWS_HEAD = 0  # gathered RM rows, NOC0 reader's half (aliased with CB_ROWS_TAIL: one allocation)
CB_ROWS_TAIL = 1  # gathered RM rows, NOC1 reader's half
CB_OUTPUT_TILES = 2  # the output shard (backed on the output tensor)
CB_GATHER_IDS = 3  # the whole gather_index, one copy a core (multicast target, read by both readers)
IDS_READY_SEMAPHORE_ID = 0  # rounds of this core's row's ids landed on this core (set by the row sender's multicast)


def _config_key(config):
    if config is None:
        config = default_compute_kernel_config()
    return (
        str(config.math_fidelity),
        bool(config.fp32_dest_acc_en),
        bool(config.dst_full_sync_en),
        bool(config.math_approx_mode),
    )


def _compute_config(key):
    config = default_compute_kernel_config()
    fidelity = {
        str(f): f
        for f in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4)
    }
    config.math_fidelity = fidelity.get(key[0], ttnn.MathFidelity.LoFi)
    config.fp32_dest_acc_en = key[1]
    config.dst_full_sync_en = key[2]
    config.math_approx_mode = key[3]
    return config


def _mcast_row_x(device):
    """x range (column 0, column GRID_X - 1) of a grid row's multicast, in NoC coordinates; the row may span
    non-worker columns, the hardware delivers to the worker cores only."""
    lo = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    hi = device.worker_core_from_logical_core(ttnn.CoreCoord(GRID_X - 1, 0))
    return (lo.x, hi.x)


@lru_cache(maxsize=None)
def _static_parts(p: Plan, index_accessor_ct: tuple, config_key: tuple, mcast_row_x: tuple):
    grid = _grid()
    w = p.block_width_tiles
    rows_cb = ttnn.CBDescriptor(
        total_size=_rows_cb_bytes(p),
        core_ranges=grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            for index in (CB_ROWS_HEAD, CB_ROWS_TAIL)
        ],
    )
    ids_cb = ttnn.CBDescriptor(
        total_size=_ids_cb_bytes(p),
        core_ranges=grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_GATHER_IDS, data_format=ttnn.uint32, page_size=IDS_ALIGN)
        ],
    )
    ids_ready = ttnn.SemaphoreDescriptor(id=IDS_READY_SEMAPHORE_ID, core_ranges=grid, initial_value=0)
    common_named = [
        ("block_width_tiles", w),
        ("num_shards", p.num_shards),
        ("num_cores", NUM_CORES),
        ("grid_x", GRID_X),
    ]
    # Reader groups: one split for the grid, or (SENDER_ROW_SHED) grid column 0 -- the ids senders -- on its own split.
    if p.sender_rows_noc0 == p.rows_noc0:
        groups = [(grid, p.rows_noc0)]
    else:
        column = lambda x0, x1: ttnn.CoreRangeSet(  # noqa: E731
            [ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, GRID_Y - 1))]
        )
        groups = [(column(0, 0), p.sender_rows_noc0), (column(1, GRID_X - 1), p.rows_noc0)]
    readers = []
    for (core_ranges, rows_noc0), (half, (source, processor, noc)) in itertools.product(
        groups,
        enumerate(
            (
                ("nomic_moe_dispatch_reader.cpp", ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                ("nomic_moe_dispatch_reader_noc1.cpp", ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            )
        ),
    ):
        row_begin, row_end = p.row_ranges(rows_noc0)[half]
        named = common_named + [
            ("cb_rows", (CB_ROWS_HEAD, CB_ROWS_TAIL)[half]),
            ("cb_gather_ids", CB_GATHER_IDS),
            ("row_begin", row_begin),
            ("row_end", row_end),
            ("piece_bytes", p.piece_bytes),
            ("table_page_bytes", ROW_BYTES),
            ("pieces_per_row", p.pieces_per_row),
            ("row_slots", p.row_slots),
            ("reads_in_flight", p.reads_in_flight),
            ("ids_align", IDS_ALIGN),
            ("ids_total_bytes", p.rows * ID_BYTES),
            ("ids_lookahead", p.ids_lookahead),
            ("ids_ready_sem", IDS_READY_SEMAPHORE_ID),
            ("ids_sender_kernel", int(half == p.ids_sender_noc)),
            # A multicast rectangle starts at its NoC's low corner: column 0 on NOC0, column GRID_X - 1 on NOC1.
            ("mcast_start_x", mcast_row_x[half]),
            ("mcast_end_x", mcast_row_x[1 - half]),
            ("row_stagger", p.row_stagger),
            ("row_strides", p.row_strides),
            ("start_spread_cycles", p.start_spread_cycles),
            ("row_split_path", int(p.row_split_path)),
            ("row_split_slack", p.row_split_slack),
            ("reader_half", half),
            ("rows_noc0", rows_noc0),
        ]
        readers.append((source, processor, noc, named, core_ranges))
    compute_named = common_named + [
        ("cb_rows_head", CB_ROWS_HEAD),
        ("cb_rows_tail", CB_ROWS_TAIL),
        ("cb_output_tiles", CB_OUTPUT_TILES),
    ]
    return grid, rows_cb, ids_cb, ids_ready, readers, compute_named, list(index_accessor_ct)


def create_program_descriptor(p: Plan, table, gather_index, output, compute_kernel_config=None):
    key = _config_key(compute_kernel_config)
    index_ct = tuple(ttnn.TensorAccessorArgs(gather_index).get_compile_time_args())
    grid, rows_cb, ids_cb, ids_ready, readers, compute_named, index_ct_list = _static_parts(
        p, index_ct, key, _mcast_row_x(table.device())
    )
    addresses = [table.buffer_address(), gather_index.buffer_address()]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=core_ranges,
            compile_time_args=index_ct_list,
            named_compile_time_args=named,
            common_runtime_args=addresses,
            config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
        )
        for source, processor, noc, named, core_ranges in readers
    ]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "nomic_moe_dispatch_compute.cpp"),
            core_ranges=grid,
            compile_time_args=[],
            named_compile_time_args=compute_named,
            config=_compute_config(key),
        )
    )
    output_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output, core_ranges=grid)
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[ids_ready], cbs=[rows_cb, output_cb, ids_cb])
