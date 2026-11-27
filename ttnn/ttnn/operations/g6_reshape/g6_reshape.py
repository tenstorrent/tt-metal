# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshape: torch.reshape on the logical values, laid out again in the input's own layout.

One device program per call. Values move bit for bit between the valid face-row segments of the
input and output tile tensors; no compute engine, no host read, no second device tensor.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_reshape_program_descriptor as _desc


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is a 1-tuple around {"input_shape", "shape", "dtype", "layout"}
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
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, shape):
    scenario = {
        "input_shape": tuple(input_tensor.shape),
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_reshape: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_reshape: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Host-side shape resolution
# ---------------------------------------------------------------------------

_MAX_ELEMENTS = 2**32


def _resolve_shape(input_tensor, shape):
    """`shape` with its -1 resolved, or ValueError for anything torch.reshape would refuse."""
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"g6_reshape: shape must be a list or tuple, got {type(shape).__name__}")
    if not 1 <= len(shape) <= _desc.MAX_RANK:
        raise ValueError(f"g6_reshape: output rank must be 1..{_desc.MAX_RANK}, got {len(shape)}")
    in_shape = tuple(int(s) for s in input_tensor.shape)
    if not 1 <= len(in_shape) <= _desc.MAX_RANK:
        raise ValueError(f"g6_reshape: input rank must be 1..{_desc.MAX_RANK}, got {len(in_shape)}")
    if any(s < 1 for s in in_shape):
        raise ValueError(f"g6_reshape: input dims must be >= 1, got {list(in_shape)}")
    for s in shape:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"g6_reshape: shape entries must be int, got {s!r}")
        if s == 0 or s < -1:
            raise ValueError(f"g6_reshape: shape entries must be >= 1 or -1, got {list(shape)}")
    n = math.prod(in_shape)
    inferred = [i for i, s in enumerate(shape) if s == -1]
    if len(inferred) > 1:
        raise ValueError(f"g6_reshape: at most one -1 entry, got {list(shape)}")
    resolved = list(shape)
    if inferred:
        known = math.prod(s for s in shape if s != -1)
        if n % known:
            raise ValueError(f"g6_reshape: cannot infer -1 in {list(shape)} for {n} elements")
        resolved[inferred[0]] = n // known
    if math.prod(resolved) != n:
        raise ValueError(f"g6_reshape: shape {list(shape)} has {math.prod(resolved)} elements, input has {n}")
    if n >= _MAX_ELEMENTS:
        raise ValueError(f"g6_reshape: {n} elements exceed the 32-bit index width")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshape: input must be on device")
    mc = input_tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"g6_reshape: input must be DRAM interleaved, got {mc}")
    return tuple(resolved)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_reshape(input_tensor: ttnn.Tensor, shape) -> ttnn.Tensor:
    validate(input_tensor, shape)
    out_shape = _resolve_shape(input_tensor, shape)

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = _desc.get_program_descriptor(input_tensor, output, out_shape)
    return ttnn.generic_op([input_tensor, output], descriptor)


def g6_reshape_plan(input_tensor: ttnn.Tensor, shape) -> dict:
    """The plan a g6_reshape call with these arguments selects (host only; dispatches nothing)."""
    validate(input_tensor, shape)
    out_shape = _resolve_shape(input_tensor, shape)
    return _desc.public_plan(_desc.get_plan(input_tensor, out_shape))
