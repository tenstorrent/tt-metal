# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reshape: `torch.reshape` on a device tensor (eval/specs/reshape/spec.py).

The output has the resolved new shape and the input's dtype and layout, and lives in a new interleaved buffer of
the input's buffer type (rule R7). Every call is one `ttnn.generic_op` dispatch of one of two ProgramSpecs
(reshape_program_artifacts.py): `page_identity` copies pages one-to-one, `tile_row_regroup` moves rows between
tiles when H changes in tile layout (rule R4).
"""

from __future__ import annotations

import math

import ttnn

from eval.memory import memory_axes
from eval.specs.reshape.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .reshape_program_artifacts import NOC_MAX_BURST_SIZE, TILE_ROWS, create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding":      ["interleaved"],
    "changes":       ["leading", "h"],
    "alignment":     ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":          [1, 3, 4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# Shape argument (rules R2, R3; rejections X1-X3)
# ---------------------------------------------------------------------------


def resolve_shape(input_shape, shape):
    """The new shape as a tuple of ints with a -1 resolved (R2). Accepts a list or a tuple (R3)."""
    if isinstance(shape, (str, bytes)) or not hasattr(shape, "__iter__"):
        raise ValueError(f"reshape: `shape` must be a list or tuple of ints, got {shape!r}")
    try:
        new_shape = [int(d) for d in shape]
    except (TypeError, ValueError) as e:
        raise ValueError(f"reshape: `shape` must be a list or tuple of ints, got {shape!r}") from e
    volume = math.prod(input_shape)
    if new_shape.count(-1) > 1:
        raise ValueError(f"reshape: `shape` {tuple(new_shape)} has more than one -1")  # X2
    if any(d < -1 for d in new_shape):
        raise ValueError(f"reshape: `shape` {tuple(new_shape)} has a negative size")
    if -1 in new_shape:
        rest = math.prod(d for d in new_shape if d != -1)
        if rest == 0 or volume % rest != 0:
            raise ValueError(f"reshape: no size fits the -1 of {tuple(new_shape)} for {volume} elements")  # X3
        new_shape[new_shape.index(-1)] = volume // rest
    if math.prod(new_shape) != volume:
        raise ValueError(
            f"reshape: {tuple(new_shape)} has {math.prod(new_shape)} elements, the tensor has {volume}"
        )  # X1
    return tuple(new_shape)


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, shape):
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout, **memory_axes(input_tensor)}  # R8
    entry = (tuple(input_tensor.shape), tuple(shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(entry, axes)
    return axes


def validate(input_tensor, shape):
    """Runtime gate (R6): malformed shapes raise ValueError (X1-X3); anything outside SUPPORTED / inside
    EXCLUSIONS raises the pipeline's NotImplementedError refusals."""
    resolve_shape(tuple(input_tensor.shape), shape)
    axes = _axes(input_tensor, shape)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"reshape: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"reshape: unsupported combination (refinement candidate): {exc}")
    # Mechanism caps (op_design.md, Mechanism caps).
    if input_tensor.layout == ttnn.TILE_LAYOUT and tuple(input_tensor.tile.tile_shape) != (TILE_ROWS, TILE_ROWS):
        raise NotImplementedError(f"reshape: tile shape {tuple(input_tensor.tile.tile_shape)} is not 32x32")
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT and input_tensor.buffer_page_size() > NOC_MAX_BURST_SIZE:
        raise NotImplementedError(
            f"reshape: a row-major page of {input_tensor.buffer_page_size()} B exceeds one NoC transaction"
        )
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def reshape(input_tensor: ttnn.Tensor, shape) -> ttnn.Tensor:
    validate(input_tensor, shape)
    return _reshape(input_tensor, shape)


def _reshape(input_tensor, shape, **knobs):
    """The dispatch after validate(); `knobs` override block extents (tests and refinements)."""
    in_shape = tuple(input_tensor.shape)
    new_shape = resolve_shape(in_shape, shape)
    output = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(new_shape, input_tensor.dtype, input_tensor.layout, input_tensor.memory_config()),
        input_tensor.device(),
    )  # R7: a new interleaved buffer of the input's buffer type
    _, _, spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, in_shape, new_shape, **knobs)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
