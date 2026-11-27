# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_unsqueeze: append a last axis of extent 1 to a one-row tensor, in one device program.

    out[i_0, ..., i_{r-1}, 0] = x[i_0, ..., i_{r-1}]

On tiles, row 0 of input tile t becomes column 0 of output tile t, bit for bit.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_unsqueeze_program_descriptor as _descriptor
from .g6_unsqueeze_program_descriptor import _plan, create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# `inputs` is a 1-tuple around the scenario dict {"input_shape", "dim", "dtype", "layout"}.


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
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    # Per-core L1 is clamped to L1_BUDGET_BYTES by _plan() for every W.
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor, dim):
    return {
        "input_shape": tuple(input_tensor.shape),
        "dim": dim,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }


def validate(input_tensor, dim):
    scenario = _scenario(input_tensor, dim)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_unsqueeze: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_unsqueeze: unsupported combination (refinement candidate): {exc}")

    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError("g6_unsqueeze: input_tensor must be on device")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"g6_unsqueeze: dim must be an int, got {type(dim).__name__}")
    shape = tuple(input_tensor.shape)
    rank = len(shape)
    if not 1 <= rank <= 3:
        raise ValueError(f"g6_unsqueeze: input rank must be 1 to 3, got {rank}")
    if not -(rank + 1) <= dim <= rank:
        raise ValueError(f"g6_unsqueeze: dim={dim} outside [{-(rank + 1)}, {rank}] for rank {rank}")
    position = dim + rank + 1 if dim < 0 else dim
    if position != rank:
        raise ValueError(f"g6_unsqueeze: dim={dim} must put the new axis last ({rank} or -1)")
    if any(d == 0 for d in shape):
        raise ValueError(f"g6_unsqueeze: input {list(shape)} has no elements")
    if any(d != 1 for d in shape[:-1]):
        raise ValueError(f"g6_unsqueeze: every dim but the last must be 1, got {list(shape)}")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(
            f"g6_unsqueeze: input must be DRAM interleaved, got "
            f"{memory_config.buffer_type}/{memory_config.memory_layout}"
        )
    if tuple(input_tensor.tile.tile_shape) != (32, 32):
        raise ValueError(f"g6_unsqueeze: input tile must be 32x32, got {tuple(input_tensor.tile.tile_shape)}")


# ---------------------------------------------------------------------------
# Plan reporting and program cache
# ---------------------------------------------------------------------------


def plan_for(input_tensor, max_cores=None):
    """The plan a call on `input_tensor` selects: regime, core rectangle, per-core role and work,
    block shape and L1 per core."""
    plan = _plan(input_tensor, max_cores)
    core_ranges = plan.core_range_set
    return {
        "regime": plan.regime,
        "rectangle": (plan.rect_cols, plan.rect_rows),
        "core_ranges": core_ranges,
        "num_core_ranges": len(core_ranges.ranges()),
        "cores": [
            {"core": (c.x, c.y), "role": c.role, "start_tile": c.start_tile, "core_w_tiles": c.core_w_tiles}
            for c in plan.cores
        ],
        "tensor_w_tiles": plan.tensor_w_tiles,
        "block_w_tiles": plan.block_w_tiles,
        "staging_depth_blocks": plan.staging_depth_blocks,
        "l1_bytes_per_core": plan.l1_bytes_per_core,
    }


# (program_descriptor, mover_kernel) per program-shaping key. ProgramDescriptor.kernels is a
# by-value property, so the kernel is kept beside the descriptor and reassigned on every call.
_PROGRAM_CACHE = {}


def _cache_key(input_tensor, device):
    memory_config = input_tensor.memory_config()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        tuple(input_tensor.tile.tile_shape),
        memory_config.buffer_type,
        memory_config.memory_layout,
        device.id(),
        (grid.x, grid.y),
        _descriptor.MAX_CORES,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_unsqueeze(input_tensor: ttnn.Tensor, dim: int) -> ttnn.Tensor:
    validate(input_tensor, dim)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape) + [1]),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, device)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        cached = create_program_descriptor(_plan(input_tensor), input_tensor, output_tensor)
        _PROGRAM_CACHE[key] = cached
    program_descriptor, mover_kernel = cached
    mover_kernel.common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    program_descriptor.kernels = [mover_kernel]

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
