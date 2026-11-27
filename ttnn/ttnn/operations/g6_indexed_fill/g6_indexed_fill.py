# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_indexed_fill: input_tensor with the batches batch_ids names replaced by value_tensor's batches.

    y[b, r, w] = s[i, r, w]   if b == ids[i] < B for some i (the largest such i)
    y[b, r, w] = x[b, r, w]   otherwise

batch_ids is read on the device in every run (unsigned; an entry >= B writes nothing). One device
program, one data-movement kernel per work core; see op_design.md.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_indexed_fill_program_descriptor import plan_for, program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """input_tensor [B, d1, d2, W] read as the plane [d2, W]; W misalignment takes precedence."""
    _, input_shape, _ = inputs
    d2, W = int(input_shape[-2]), int(input_shape[-1])
    if W % 32 != 0:
        return "w_non_aligned"
    if d2 % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.uint32],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    # ROW_MAJOR pages carry no tile padding, so the kernel never branches on alignment; refusing a
    # value here would refuse valid shapes on the host (spec rule: every B, K, d1, d2, W).
    "alignment": ["h_non_aligned", "w_non_aligned", "tile_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(tensor):
    config = tensor.memory_config()
    return config.buffer_type == ttnn.BufferType.DRAM and config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def validate(batch_ids, input_tensor, value_tensor, *, dim=0):
    """Refuse every call the kernel cannot serve. No check reads a value of batch_ids."""
    if dim != 0:
        raise ValueError(f"g6_indexed_fill: dim must be 0, got {dim!r}")

    # Ranks first: the tagger reads input_tensor.shape[2:].
    if len(batch_ids.shape) != 1:
        raise ValueError(f"g6_indexed_fill: batch_ids must have rank 1, got shape {list(batch_ids.shape)}")
    if len(input_tensor.shape) != 4:
        raise ValueError(f"g6_indexed_fill: input_tensor must have rank 4, got shape {list(input_tensor.shape)}")
    if len(value_tensor.shape) != 4:
        raise ValueError(f"g6_indexed_fill: value_tensor must have rank 4, got shape {list(value_tensor.shape)}")

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    inputs = (tuple(batch_ids.shape), tuple(input_tensor.shape), tuple(value_tensor.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_indexed_fill: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_indexed_fill: unsupported combination (refinement candidate): {exc}")

    for name, tensor in (("batch_ids", batch_ids), ("value_tensor", value_tensor)):
        if tensor.dtype != ttnn.uint32:
            raise ValueError(f"g6_indexed_fill: {name} must be uint32, got {tensor.dtype}")
        if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"g6_indexed_fill: {name} must be ROW_MAJOR, got {tensor.layout}")

    K = int(batch_ids.shape[0])
    if K < 1:
        raise ValueError("g6_indexed_fill: batch_ids must hold at least one entry")
    if int(value_tensor.shape[0]) != K:
        raise ValueError(
            f"g6_indexed_fill: value_tensor has {int(value_tensor.shape[0])} batches, batch_ids has {K} entries"
        )
    input_shape, value_shape = tuple(input_tensor.shape), tuple(value_tensor.shape)
    if value_shape[1:] != input_shape[1:]:
        raise ValueError(
            f"g6_indexed_fill: value_tensor shape {list(value_tensor.shape)} does not match "
            f"input_tensor shape {list(input_tensor.shape)} beyond dim 0"
        )
    if any(int(v) < 1 for v in input_tensor.shape):
        raise ValueError(f"g6_indexed_fill: every extent of input_tensor must be >= 1, got {list(input_tensor.shape)}")

    tensors = (("batch_ids", batch_ids), ("input_tensor", input_tensor), ("value_tensor", value_tensor))
    for name, tensor in tensors:
        if not ttnn.is_tensor_storage_on_device(tensor):
            raise ValueError(f"g6_indexed_fill: {name} must be on the device")
    device_id = input_tensor.device().id()
    for name, tensor in tensors:
        if tensor.device().id() != device_id:
            raise ValueError(f"g6_indexed_fill: {name} is on another device than input_tensor")
        if not _is_dram_interleaved(tensor):
            raise ValueError(f"g6_indexed_fill: {name} must be DRAM interleaved, got {tensor.memory_config()}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_indexed_fill(
    batch_ids: ttnn.Tensor,
    input_tensor: ttnn.Tensor,
    value_tensor: ttnn.Tensor,
    *,
    dim: int = 0,
) -> ttnn.Tensor:
    validate(batch_ids, input_tensor, value_tensor, dim=dim)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = program_descriptor(batch_ids, input_tensor, value_tensor, output_tensor)
    return ttnn.generic_op([batch_ids, input_tensor, value_tensor, output_tensor], descriptor)


def describe_plan(batch_ids, input_tensor, value_tensor, *, dim=0):
    """The plan the same call would dispatch (regime, cores, blocks, L1 use, per-core roles)."""
    validate(batch_ids, input_tensor, value_tensor, dim=dim)
    return plan_for(batch_ids, input_tensor, value_tensor).describe()
