# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize: uint32 TILE -> ROW_MAJOR, bit-identical, one device program per call.

The reorder is a NoC gather of 64 B face-row pieces into row order in L1, followed by whole-page
writes (row_gather regime, op_design.md). No compute kernel touches the values.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_untilize_program_descriptor as pd

_OP = "g6_untilize"


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs = (scenario,), scenario = {"input_shape", "dtype", "layout"}
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.uint32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor):
    return {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }


def validate(input_tensor, *, use_multicore=True, sub_core_grids=None):
    if use_multicore is not True:
        raise ValueError(f"{_OP}: use_multicore must be True, got {use_multicore!r}")
    if sub_core_grids is not None:
        raise ValueError(f"{_OP}: sub_core_grids must be None, got {sub_core_grids!r}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError(f"{_OP}: input_tensor must be a device tensor (storage {input_tensor.storage_type()!r})")
    if len(input_tensor.shape) != 4:
        raise ValueError(f"{_OP}: input_tensor must have rank 4, got shape {list(input_tensor.shape)}")
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"{_OP}: input_tensor must be DRAM interleaved, got {mc}")

    scenario = _scenario(input_tensor)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{_OP}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{_OP}: unsupported combination (refinement candidate): {exc}")

    # Not an axis: the face-row offsets assume the standard 32x32 tile of four 16x16 faces.
    tile_shape = tuple(input_tensor.tile.tile_shape)
    if tile_shape != (pd.TILE_HEIGHT, pd.TILE_WIDTH):
        raise ValueError(f"{_OP}: tile shape {tile_shape} is not ({pd.TILE_HEIGHT}, {pd.TILE_WIDTH})")


# ---------------------------------------------------------------------------
# Program cache (BB-07): one built descriptor per program-shaping key.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _cache_key(input_tensor, grid):
    mc = input_tensor.memory_config()
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        mc.memory_layout,
        mc.buffer_type,
        grid.x,
        grid.y,
        # knobs, so that turning one at runtime builds a new program
        pd.BLOCK_TILE_COLS_REQ,
        pd.SCRATCH_DEPTH,
        pd.CORE_BUDGET_REQ,
        pd.CORE_ORIGIN,
        pd.KERNEL_PROCESSOR,
        pd.KERNEL_NOC,
    )


def _plan_for(input_tensor):
    grid = input_tensor.device().compute_with_storage_grid_size()
    key = _cache_key(input_tensor, grid)
    entry = _PROGRAM_CACHE.get(key)
    plan = (
        entry.plan
        if entry is not None
        else pd.make_plan(input_tensor.shape, input_tensor.element_size(), grid.x, grid.y)
    )
    return key, entry, plan


def describe_plan(input_tensor) -> dict:
    """The plan the entry point runs for this input (regime, cores, block knobs, per-core role)."""
    _, _, plan = _plan_for(input_tensor)
    return pd.describe(plan)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_untilize(input_tensor, *, use_multicore=True, sub_core_grids=None):
    validate(input_tensor, use_multicore=use_multicore, sub_core_grids=sub_core_grids)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    key, entry, plan = _plan_for(input_tensor)
    if entry is None:
        entry = pd.create_program(input_tensor, output_tensor, plan)
        _PROGRAM_CACHE[key] = entry
    descriptor = entry.bind(input_tensor.buffer_address(), output_tensor.buffer_address())
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
