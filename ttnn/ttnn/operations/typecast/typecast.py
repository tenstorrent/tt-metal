# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""typecast(input_tensor, dtype): every element of the input converted to `dtype`.

Registry model (eval/op_template.py): INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS,
validate(), and the entry point, which validates and then dispatches exactly one Metal 2.0 ProgramSpec
through ttnn.generic_op (typecast_program_artifacts.py).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.typecast.spec import AXES

from .typecast_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions (alignment, rank)
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — the specification's PHASE0
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16, ttnn.float32],
    "target_dtype":  [ttnn.bfloat16, ttnn.float32],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding":      ["interleaved"],
    "alignment":     ["tile_aligned"],
    "rank":          [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, dtype):
    axes = {
        "dtype": input_tensor.dtype,
        "target_dtype": dtype,
        "layout": input_tensor.layout,
        **memory_axes(input_tensor),  # R7: memory_config, sharding
    }
    inputs = (tuple(input_tensor.shape),)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(input_tensor, dtype):
    # REJECTIONS X1: a malformed call, refused before any axis is derived from it.
    if not isinstance(dtype, ttnn.DataType):
        raise TypeError(f"typecast: dtype must be a ttnn.DataType, got {type(dtype).__name__} {dtype!r}")
    if not isinstance(input_tensor, ttnn.Tensor):
        raise TypeError(f"typecast: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")

    axes = _axes(input_tensor, dtype)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"typecast: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"typecast: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_layout(input_tensor, dtype):
    """The input's layout, except that a block-float output is always in tiles (spec OUTPUT)."""
    return ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else input_tensor.layout


def typecast(input_tensor: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
    validate(input_tensor, dtype)

    device = input_tensor.device()
    # R6: the output lives in the input's buffer, interleaved. A same-dtype call still gets a fresh tensor and
    # the same single dispatch (the conversion is then a copy).
    out_spec = ttnn.TensorSpec(
        input_tensor.shape,
        dtype,
        _output_layout(input_tensor, dtype),
        buffer_type=input_tensor.memory_config().buffer_type,
    )
    output = ttnn.allocate_tensor_on_device(out_spec, device)

    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
