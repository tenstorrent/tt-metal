# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_copy(input_tensor, output_tensor) -> output_tensor.

Writes every logical element of `input_tensor` into the caller's `output_tensor` (same shape, dtype,
layout, DRAM interleaved placement) bit for bit, in one device program, and returns `output_tensor`
itself. Realized as a byte-exact copy of every buffer page: both tensors share the page -> bank map,
so source tile padding lands only in destination tile padding.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_copy_program_descriptor as _descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    if w % 32:
        return "w_non_aligned"
    if h % 32:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    {"dtype": ttnn.float32, "layout": ttnn.ROW_MAJOR_LAYOUT},
    {"dtype": ttnn.int32, "layout": ttnn.TILE_LAYOUT},
    {"dtype": ttnn.uint32, "layout": ttnn.TILE_LAYOUT},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _validate_request(input_tensor, output_tensor):
    """Malformed requests (not support refusals): ValueError."""
    for name, t in (("input_tensor", input_tensor), ("output_tensor", output_tensor)):
        if not ttnn.is_tensor_storage_on_device(t):
            raise ValueError(f"g6_copy: {name} must be on a device")
    rank = len(input_tensor.shape)
    if not 1 <= rank <= 4:
        raise ValueError(f"g6_copy: rank must be 1 to 4, got {rank}")
    if tuple(input_tensor.shape) != tuple(output_tensor.shape):
        raise ValueError(
            f"g6_copy: shapes differ: input {tuple(input_tensor.shape)}, output {tuple(output_tensor.shape)}"
        )
    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(f"g6_copy: dtypes differ: input {input_tensor.dtype}, output {output_tensor.dtype}")
    if input_tensor.layout != output_tensor.layout:
        raise ValueError(f"g6_copy: layouts differ: input {input_tensor.layout}, output {output_tensor.layout}")
    in_memory = input_tensor.memory_config()
    if in_memory != output_tensor.memory_config():
        raise ValueError(f"g6_copy: memory configs differ: input {in_memory}, output {output_tensor.memory_config()}")
    if in_memory.buffer_type != ttnn.BufferType.DRAM or in_memory.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"g6_copy: tensors must be DRAM interleaved, got {in_memory}")


def validate(input_tensor, output_tensor):
    _validate_request(input_tensor, output_tensor)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_copy: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_copy: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_copy(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(input_tensor, output_tensor)
    descriptor = _descriptor.get_program_descriptor(input_tensor, output_tensor)
    ttnn.generic_op([input_tensor, output_tensor], descriptor)
    return output_tensor
