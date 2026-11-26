# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_repeat program descriptor: regime 1, column_split_row_broadcast.

Out tile columns are split contiguously over the grid. On every active core one
data-movement kernel loads the row-0 faces of its x tiles straight into the
out-tile slots of cb_repeat_tiles, replicates row 0 in place by NoC loopback, and
writes each out page once.

The knobs below are the single source of every block extent, buffer depth and
core count; the CB size and every kernel trip count derive from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Knobs (read at call time, so a test or a tuning pass can monkeypatch them)
# ---------------------------------------------------------------------------

# Slot sets of cb_repeat_tiles; > 1 lets block b's writes drain while block b+1 loads.
REPEAT_BUFFER_DEPTH = 1
# L1 the model leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1347776
# Upper bound on block_width_tiles; None means the L1 cap alone.
MAX_BLOCK_WIDTH_TILES = None
# Upper bound on active cores; None means the full compute-with-storage grid.
MAX_NUM_CORES = None

REGIME = "column_split_row_broadcast"

CB_REPEAT_TILES = 0

# NoC L1-to-L1 transfers need both ends 16 B aligned.
L1_NOC_ALIGNMENT_BYTES = 16


@dataclass(frozen=True)
class RepeatPlan:
    regime: str
    num_cores: int
    all_cores: object
    core_ranges: tuple  # ((CoreCoord, start_tile_col, core_w_tiles), ...)
    block_width_tiles: int
    max_block_width_tiles: int
    repeat_buffer_depth: int
    tensor_out_wt_tiles: int
    tensor_out_ht_tiles: int
    rows_to_fill: int
    out_page_size: int
    x_page_size: int
    face_row_bytes: int
    face_bytes: int
    face_cols: int
    face_h: int
    row0_faces_bytes: int


def _div_up(a, b):
    return (a + b - 1) // b


def _out_shape(input_tensor, reps):
    return [int(d) * int(r) for d, r in zip(input_tensor.shape, reps)]


def _device_key(device):
    try:
        return device.id()
    except Exception:  # noqa: BLE001 — a device without id() falls back to object identity
        return id(device)


def _cache_key(input_tensor, reps):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    mc = input_tensor.memory_config()
    return (
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(r) for r in reps),
        input_tensor.dtype,
        input_tensor.layout,
        tuple(input_tensor.tile.tile_shape),
        tuple(input_tensor.tile.face_shape),
        mc.memory_layout,
        mc.buffer_type,
        _device_key(device),
        (grid.x, grid.y),
        REPEAT_BUFFER_DEPTH,
        L1_BUDGET_BYTES,
        MAX_BLOCK_WIDTH_TILES,
        MAX_NUM_CORES,
    )


def _make_plan(input_tensor, reps) -> RepeatPlan:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()

    tile_h, tile_w = (int(v) for v in input_tensor.tile.tile_shape)
    face_h, face_w = (int(v) for v in input_tensor.tile.face_shape)
    element_size = int(input_tensor.element_size())
    face_row_bytes = face_w * element_size
    face_bytes = face_h * face_row_bytes
    face_cols = tile_w // face_w
    row0_faces_bytes = face_cols * face_bytes
    assert face_row_bytes % L1_NOC_ALIGNMENT_BYTES == 0, f"face row of {face_row_bytes} B breaks L1 NoC alignment"

    # x and out share dtype, layout and tile, so one page size describes both.
    page_size = int(input_tensor.buffer_aligned_page_size())
    assert page_size % ttnn.get_dram_alignment() == 0, "slot stride must keep the DRAM-read alignment"
    assert row0_faces_bytes <= page_size

    out_shape = _out_shape(input_tensor, reps)
    h_out, w_out = out_shape[-2], out_shape[-1]
    tensor_out_wt_tiles = _div_up(w_out, tile_w)
    tensor_out_ht_tiles = out_shape[0] * out_shape[1] * _div_up(h_out, tile_h)
    rows_to_fill = min(h_out, tile_h)

    grid_cores = grid.x * grid.y
    max_num_cores = grid_cores if MAX_NUM_CORES is None else max(1, min(int(MAX_NUM_CORES), grid_cores))
    core_set = ttnn.num_cores_to_corerangeset(min(tensor_out_wt_tiles, max_num_cores), grid, row_wise=True)
    num_cores, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        core_set, tensor_out_wt_tiles, row_wise=True
    )

    max_block_width_tiles = L1_BUDGET_BYTES // (REPEAT_BUFFER_DEPTH * page_size)
    if MAX_BLOCK_WIDTH_TILES is not None:
        max_block_width_tiles = min(max_block_width_tiles, int(MAX_BLOCK_WIDTH_TILES))
    if max_block_width_tiles < 1:
        raise ValueError(
            f"g6_repeat: one slot set of {REPEAT_BUFFER_DEPTH} x {page_size} B exceeds the L1 budget "
            f"of {L1_BUDGET_BYTES} B"
        )
    block_width_tiles = min(per_core_1, max_block_width_tiles)

    core_ranges = []
    start_tile_col = 0
    for group, core_w_tiles in ((group_1, per_core_1), (group_2, per_core_2)):
        if core_w_tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            core_ranges.append((core, start_tile_col, core_w_tiles))
            start_tile_col += core_w_tiles
    assert start_tile_col == tensor_out_wt_tiles

    return RepeatPlan(
        regime=REGIME,
        num_cores=num_cores,
        all_cores=all_cores,
        core_ranges=tuple(core_ranges),
        block_width_tiles=block_width_tiles,
        max_block_width_tiles=max_block_width_tiles,
        repeat_buffer_depth=REPEAT_BUFFER_DEPTH,
        tensor_out_wt_tiles=tensor_out_wt_tiles,
        tensor_out_ht_tiles=tensor_out_ht_tiles,
        rows_to_fill=rows_to_fill,
        out_page_size=page_size,
        x_page_size=page_size,
        face_row_bytes=face_row_bytes,
        face_bytes=face_bytes,
        face_cols=face_cols,
        face_h=face_h,
        row0_faces_bytes=row0_faces_bytes,
    )


def _build_descriptor(plan: RepeatPlan, input_tensor, output_tensor) -> ttnn.ProgramDescriptor:
    cb_repeat_tiles = ttnn.CBDescriptor(
        total_size=plan.repeat_buffer_depth * plan.block_width_tiles * plan.out_page_size,
        core_ranges=plan.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_REPEAT_TILES,
                data_format=output_tensor.dtype,
                page_size=plan.out_page_size,
            )
        ],
    )

    ct_args = [
        CB_REPEAT_TILES,
        plan.block_width_tiles,
        plan.repeat_buffer_depth,
        plan.rows_to_fill,
        plan.tensor_out_wt_tiles,
        plan.tensor_out_ht_tiles,
        plan.out_page_size,
        plan.x_page_size,
        plan.row0_faces_bytes,
        plan.face_row_bytes,
        plan.face_bytes,
        plan.face_cols,
        plan.face_h,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for core, start_tile_col, core_w_tiles in plan.core_ranges:
        rt_args[core.x][core.y] = [start_tile_col, core_w_tiles]

    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_repeat_dataflow.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb_repeat_tiles])


_PLAN_CACHE = {}
_DESCRIPTOR_CACHE = {}


def _get_plan(input_tensor, reps, key=None):
    key = key if key is not None else _cache_key(input_tensor, reps)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _make_plan(input_tensor, reps)
        _PLAN_CACHE[key] = plan
    return plan


def get_program(input_tensor, output_tensor, reps) -> ttnn.ProgramDescriptor:
    """The cached descriptor for this call, with the x / out addresses refreshed.

    The key holds every value that shapes the program; buffer addresses are the
    only per-call values and live in the common runtime args.
    """
    key = _cache_key(input_tensor, reps)
    desc = _DESCRIPTOR_CACHE.get(key)
    if desc is None:
        desc = _build_descriptor(_get_plan(input_tensor, reps, key), input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[key] = desc
    else:
        common_args = desc.kernels[0].common_runtime_args
        common_args[0] = input_tensor.buffer_address()
        common_args[1] = output_tensor.buffer_address()
    return desc


def plan_report(input_tensor, reps) -> dict:
    plan = _get_plan(input_tensor, reps)
    return {
        "regime": plan.regime,
        "num_cores": plan.num_cores,
        "block_width_tiles": plan.block_width_tiles,
        "repeat_buffer_depth": plan.repeat_buffer_depth,
        "num_blocks": max(_div_up(w, plan.block_width_tiles) for _, _, w in plan.core_ranges),
        "cores": [
            {"core": (core.x, core.y), "role": "repeat", "start_tile_col": start, "core_w_tiles": w}
            for core, start, w in plan.core_ranges
        ],
    }
