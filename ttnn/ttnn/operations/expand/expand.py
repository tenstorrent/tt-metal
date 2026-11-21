# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""expand: `torch.Tensor.expand`, made a tensor of its own (the op copies).

Registry-model op file: INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS, validate(), and the
public entry point. The device program (reader + writer, no compute kernel) is built in
expand_program_artifacts.py and dispatched as one `ttnn.generic_op`.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes  # noqa: E402  (R7: the placement axes, named as the suite names them)
from eval.specs.expand.spec import AXES  # noqa: E402

from .expand_program_artifacts import create_program_artifacts


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES


# ---------------------------------------------------------------------------
# 2. SUPPORTED — the first phase: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding":      ["interleaved"],
    "changes":       ["h"],
    "alignment":     ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":          [2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Host-side contract: the call's shape arguments (REJECTIONS X1-X3, R2, R3)
# ---------------------------------------------------------------------------


def _canonical_output_shape(input_shape, output_shape):
    """`output_shape` as a list of ints (R3: list or tuple), checked against the tensor's shape (X1-X3).

    Returns the shape as asked (with its -1s): the axis functions read the -1s themselves (R2)."""
    try:
        asked = [int(d) for d in output_shape]
    except TypeError as e:
        raise ValueError(f"expand: output_shape must be a list or tuple of ints, got {output_shape!r}") from e
    shape = [int(d) for d in input_shape]
    if len(asked) < len(shape):  # X1
        raise ValueError(f"expand: output_shape {asked} has fewer entries than the tensor {shape} has dimensions")
    added = len(asked) - len(shape)
    for i, size in enumerate(asked[:added]):  # X3
        if size == -1:
            raise ValueError(f"expand: -1 at index {i} of {asked}: an added dimension needs a size")
        if size < 0:
            raise ValueError(f"expand: negative size {size} at index {i} of {asked}")
    for i, (have, want) in enumerate(zip(shape, asked[added:])):  # X2, R2
        if want == -1 or want == have:
            continue
        if want < -1 or have != 1:
            raise ValueError(
                f"expand: dimension {i} of the tensor {shape} has size {have} and cannot become {want} "
                f"(only a dimension of size 1 expands)"
            )
    if 0 in asked:
        raise NotImplementedError(f"expand: a size of 0 in output_shape {asked} is not supported")
    return asked


def _result_shape(input_shape, asked):
    """output_shape with every -1 replaced by the tensor's size (dimensions matched from the back)."""
    shape = [int(d) for d in input_shape]
    added = len(asked) - len(shape)
    return [shape[i - added] if i >= added and size == -1 else size for i, size in enumerate(asked)]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, output_shape):
    """Raise for a malformed call (ValueError, REJECTIONS) and for a call outside SUPPORTED / inside EXCLUSIONS
    (UnsupportedAxisValue / ExcludedCell, both NotImplementedError: R8). Returns the canonical output shape."""
    asked = _canonical_output_shape(input_tensor.shape, output_shape)
    axes = {
        "dtype":  input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory_axes(input_tensor),  # R7: "memory_config", "sharding"
    }
    inputs = (tuple(int(d) for d in input_tensor.shape), tuple(asked))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    # Every axis outside SUPPORTED is named in the one refusal (R7: a sharded L1 input is refused on `sharding`,
    # not only on its buffer).
    outside = [f"{axis}={axes[axis]!r} not in SUPPORTED {allowed}" for axis, allowed in SUPPORTED.items()
               if axes[axis] not in allowed]
    if outside:
        raise UnsupportedAxisValue("expand: " + "; ".join(outside))
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"expand: unsupported combination (refinement candidate): {exc}")
    return asked


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def expand(input_tensor: ttnn.Tensor, output_shape) -> ttnn.Tensor:
    asked = validate(input_tensor, output_shape)
    result_shape = _result_shape(input_tensor.shape, asked)

    # R6: the output lives where the input does (its buffer type, interleaved).
    output_spec = ttnn.TensorSpec(
        ttnn.Shape(result_shape), input_tensor.dtype, input_tensor.layout, memory_config=input_tensor.memory_config()
    )
    output = ttnn.allocate_tensor_on_device(output_spec, input_tensor.device())

    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, result_shape)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
