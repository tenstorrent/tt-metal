# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: x.expand(*sizes).contiguous() for x of shape (d_0, d_1, 1, W), bit-exact, one device program.

Production: x (1, 1, 1, W) -> out (1, 1, W, W), out[0, 0, h, w] = x[0, 0, 0, w].
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import Plan, output_shape, program_descriptor, select_plan

__all__ = ["g6_expand", "validate", "select_plan", "Plan", "INPUT_TAGGERS", "SUPPORTED", "EXCLUSIONS"]


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Alignment of the OUTPUT shape: x has one row in every production case, so its own H is never read."""
    scenario = inputs[0]
    out = output_shape(scenario["shape"], scenario["sizes"])
    if out[-1] % 32:
        return "w_non_aligned"
    if out[-2] % 32:
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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_structure(input_tensor, sizes) -> None:
    """Contract checks that raise ValueError. They run first: the tagger needs a well-formed output shape."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_expand: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input_tensor must be on a device")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_expand: input_tensor must be DRAM interleaved, got {memory_config}")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_expand: input_tensor must have rank 4, got shape {shape}")
    try:
        sizes = [int(s) for s in sizes]
    except (TypeError, ValueError) as error:
        raise ValueError(f"g6_expand: sizes must be a list or tuple of ints, got {sizes!r}") from error
    if len(sizes) != len(shape):
        raise ValueError(f"g6_expand: sizes must have the rank of input_tensor ({len(shape)}), got {sizes}")
    for dim, size in enumerate(sizes):
        if size != -1 and size < 1:
            raise ValueError(f"g6_expand: sizes[{dim}] = {size}: each size must be -1 or >= 1")

    out = output_shape(shape, sizes)
    for dim, (in_size, out_size) in enumerate(zip(shape, out)):
        if in_size != 1 and out_size != in_size:
            raise ValueError(
                f"g6_expand: sizes[{dim}] = {out_size} differs from input size {in_size} along a dim that is not 1"
            )
    for dim in (0, 1, 3):
        if out[dim] != shape[dim]:
            raise ValueError(f"g6_expand: only dim 2 may be expanded; sizes {sizes} expands dim {dim}")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: input_tensor must have size 1 along dim 2, got shape {shape}")


def validate(input_tensor, sizes) -> None:
    _check_structure(input_tensor, sizes)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": list(input_tensor.shape), "sizes": [int(s) for s in sizes]}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    """Materialize input_tensor.expand(*sizes) as a new tensor in input_tensor's dtype, layout and memory config."""
    validate(input_tensor, sizes)

    out_shape = output_shape(list(input_tensor.shape), sizes)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor, plan = program_descriptor(input_tensor, output_tensor)
    g6_expand.last_plan = plan
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


# The plan of the most recent call (BB-08); None before the first call.
g6_expand.last_plan = None
