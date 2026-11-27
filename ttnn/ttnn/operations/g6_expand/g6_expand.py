# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: x (d_0, d_1, 1, W) broadcast along dim 2 to (d_0, d_1, H_out, W), bit-exact.

One device program per call (regime ``column_group_broadcast``, op_design.md): column groups
along grid y each read their x pages once, expand row 0 of every page into a full tile in
place, multicast the expanded tiles along their grid row, and every core of the rectangle
writes its own out rows of those columns.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import (
    REGIME,
    bind_buffer_addresses,
    create_program_descriptor,
    select_plan,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _resolved_out_shape(shape, sizes):
    return [int(d) if int(s) == -1 else int(s) for d, s in zip(shape, sizes)]


def tag_alignment(inputs, axes):
    scenario = inputs[0]
    out_shape = _resolved_out_shape(scenario["shape"], scenario["sizes"])
    if out_shape[-1] % 32:
        return "w_non_aligned"
    if out_shape[-2] % 32:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, sizes):
    shape = list(input_tensor.shape)
    sizes = list(sizes)
    if len(shape) != 4:
        raise ValueError(f"g6_expand: input must have rank 4, got shape {shape}")
    if len(sizes) != len(shape):
        raise ValueError(f"g6_expand: sizes {sizes} must have the input's rank {len(shape)}")
    for s in sizes:
        if int(s) < -1 or int(s) == 0:
            raise ValueError(f"g6_expand: sizes {sizes} must hold -1 or positive ints")
    for k, (d, s) in enumerate(zip(shape, sizes)):
        if d != 1 and int(s) not in (-1, d):
            raise ValueError(f"g6_expand: sizes[{k}]={s} differs from the input's non-unit size {d}")
        if k != 2 and d == 1 and int(s) not in (-1, 1):
            raise ValueError(f"g6_expand: only dim 2 may be expanded, got sizes[{k}]={s}")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: input dim 2 must be 1, got shape {shape}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input must be on a device")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_expand: input must be DRAM interleaved, got {memory_config}")
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        tile = input_tensor.tile
        if list(tile.tile_shape) != [32, 32] or list(tile.face_shape) != [16, 16]:
            raise ValueError(
                f"g6_expand: tile {list(tile.tile_shape)} / face {list(tile.face_shape)} must be 32x32 / 16x16"
            )

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": shape, "sizes": sizes}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Program artifact cache (BB-07): one descriptor per key; a hit only binds addresses.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _cache_key(input_tensor, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        tuple(input_tensor.padded_shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        tuple(input_tensor.tile.tile_shape),
        memory_config.buffer_type,
        memory_config.memory_layout,
        device.id(),
        (grid.x, grid.y),
    )


def _plan_for(input_tensor, out_shape):
    key = _cache_key(input_tensor, out_shape)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        plan = select_plan(input_tensor, out_shape)
        entry = {"plan": plan, "program": None}
        _PROGRAM_CACHE[key] = entry
    return entry


def _report(plan):
    keys = (
        "num_col_groups",
        "num_row_groups",
        "core_src_col_tiles",
        "core_out_row_tiles",
        "block_src_col_tiles",
        "block_out_row_tiles",
        "src_block_depth",
        "seed_rows",
        "num_cores",
        "l1_bytes_per_core",
    )
    report = {"regime": REGIME}
    report.update({k: plan[k] for k in keys})
    report["core_roles"] = dict(plan["core_roles"])
    return report


def g6_expand_plan(input_tensor: ttnn.Tensor, sizes) -> dict:
    """The plan g6_expand(input_tensor, sizes) runs (BB-08); core_roles is the role dump (BB-21)."""
    validate(input_tensor, sizes)
    out_shape = _resolved_out_shape(list(input_tensor.shape), list(sizes))
    return _report(_plan_for(input_tensor, out_shape)["plan"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    validate(input_tensor, sizes)

    out_shape = _resolved_out_shape(list(input_tensor.shape), list(sizes))
    entry = _plan_for(input_tensor, out_shape)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    if entry["program"] is None:
        entry["program"] = create_program_descriptor(input_tensor, output_tensor, entry["plan"])
    program = entry["program"]
    bind_buffer_addresses(program, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], program["descriptor"])
