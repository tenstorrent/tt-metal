# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_unsqueeze: append a size-1 axis last to a one-logical-row float32 TILE tensor in DRAM.

out[0,...,0, w, 0] = x[0,...,0, w]. One device program (ttnn.generic_op) per call; the values move
bit for bit by NoC transfers and 32-bit RISC stores only (no compute kernel).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_unsqueeze_program_descriptor import create_program_descriptor, select_plan


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the golden harness hands a scenario dict
#    {"input_shape", "dim", "dtype", "layout"}; dtype and layout are projected off it.
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
# 2. SUPPORTED (= feature_spec TARGET)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _canonical_dim(dim, rank):
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"g6_unsqueeze: dim must be an int, got {dim!r}")
    if not -(rank + 1) <= dim <= rank:
        raise ValueError(f"g6_unsqueeze: dim {dim} out of range [{-(rank + 1)}, {rank}] for rank {rank}")
    k = dim if dim >= 0 else dim + rank + 1
    if k != rank:
        raise ValueError(f"g6_unsqueeze: only the new last axis is supported (dim {rank} or -1), got dim={dim}")
    return k


def validate(input_tensor, dim):
    shape = tuple(int(s) for s in input_tensor.shape)
    rank = len(shape)

    # dim first (canonicalized; positive and negative aliases of the last position both pass).
    _canonical_dim(dim, rank)

    # Registry axes: tensor properties projected through the op's own taggers.
    scenario = {"input_shape": shape, "dim": dim, "dtype": input_tensor.dtype, "layout": input_tensor.layout}
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_unsqueeze: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_unsqueeze: unsupported combination (refinement candidate): {exc}")

    # Shape contract: one logical row, rank 1..3, non-empty.
    if not 1 <= rank <= 3:
        raise ValueError(f"g6_unsqueeze: input rank must be 1..3, got {rank} ({list(shape)})")
    if any(s != 1 for s in shape[:-1]):
        raise ValueError(f"g6_unsqueeze: every dim but the last must be 1, got {list(shape)}")
    if shape[-1] < 1:
        raise ValueError(f"g6_unsqueeze: empty input {list(shape)}")

    # Placement: device, DRAM interleaved.
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError("g6_unsqueeze: input must be on device")
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"g6_unsqueeze: input must be DRAM interleaved, got {mc.memory_layout}/{mc.buffer_type}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_unsqueeze(input_tensor, dim):
    validate(input_tensor, dim)

    out_shape = [int(s) for s in input_tensor.shape] + [1]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    plan = select_plan(input_tensor)
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
