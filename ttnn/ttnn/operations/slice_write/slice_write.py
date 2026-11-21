# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""slice_write: write a row-major source into the block `start..end` of a caller-owned
row-major destination in DRAM, in place, and return that destination.

    output[start[0]:end[0], ..., start[3]:end[3]] = input      (step 1)

Specification: eval/specs/slice_write/spec.py. Design: op_design.md (regime
`shard_owner_writes`: every core that holds a shard of the source writes its own rows
straight from the resident shard to their destination pages).
"""

from __future__ import annotations

import math

import ttnn

from eval import memory
from eval.specs.slice_write.spec import AXES, canonical_step
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .slice_write_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":          [ttnn.bfloat16],
    "layout":         [ttnn.ROW_MAJOR_LAYOUT],
    "memory_config":  ["L1"],
    "sharding":       ["height"],
    "step":           ["ones"],
    "written_region": ["outer"],
    "alignment":      ["tile_aligned", "h_non_aligned"],
    "rank":           [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# The destination's placement is a value, not an axis (spec CALL, rule R8).
_DESTINATION_PLACEMENT = {"memory_config": "DRAM", "sharding": "interleaved"}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _as_int_tuple(name, value):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"slice_write: `{name}` must be a list or tuple of ints, got {type(value).__name__}")
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"slice_write: `{name}` must hold ints, got {value!r}") from error


def _check_block(input_tensor, output_tensor, start, end, step):
    """REJECTIONS X1-X3: the arguments must describe a block of the destination that the source fills."""
    out_shape = tuple(int(d) for d in output_tensor.shape)
    in_shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(out_shape)
    for name, value in (("start", start), ("end", end), ("step", step)):
        if len(value) != rank:  # X1
            raise ValueError(f"slice_write: `{name}` has {len(value)} entries; the destination has rank {rank}")
    for d, (first, last, size) in enumerate(zip(start, end, out_shape)):
        if first < 0 or last > size or last < first:  # X2
            raise ValueError(f"slice_write: block [{first}, {last}) of dimension {d} is not inside 0..{size}")
    block = tuple(last - first for first, last in zip(start, end))
    if in_shape != block:  # X3
        raise ValueError(f"slice_write: the source's shape {list(in_shape)} is not the block's {list(block)}")
    return in_shape, out_shape


def validate(input_tensor, output_tensor, start, end, step):
    """Raise ValueError for a malformed call (REJECTIONS) and NotImplementedError for a call outside
    SUPPORTED / inside EXCLUSIONS (R9), or with a destination placed, typed or laid out otherwise (R8).
    Returns the arguments as tuples of ints."""
    raw_step = step
    start, end, step = (_as_int_tuple(n, v) for n, v in (("start", start), ("end", end), ("step", step)))
    in_shape, out_shape = _check_block(input_tensor, output_tensor, start, end, step)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory.memory_axes(input_tensor),
        "step": canonical_step(raw_step, {}),
    }
    inputs = (in_shape, out_shape, start, end)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"slice_write: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"slice_write: unsupported combination (refinement candidate): {exc}")

    # R8: the destination is DRAM interleaved, with the source's dtype and layout.
    placement = memory.memory_axes(output_tensor)
    if placement != _DESTINATION_PLACEMENT:
        raise UnsupportedAxisValue(f"slice_write: the destination must be {_DESTINATION_PLACEMENT}, got {placement}")
    if output_tensor.dtype != input_tensor.dtype or output_tensor.layout != input_tensor.layout:
        raise UnsupportedAxisValue(
            f"slice_write: the destination must have the source's dtype and layout "
            f"({input_tensor.dtype}, {input_tensor.layout}), got ({output_tensor.dtype}, {output_tensor.layout})"
        )
    if math.prod(in_shape) == 0:
        raise UnsupportedAxisValue("slice_write: a block with no elements is not supported")
    return start, end, step


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def slice_write(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, start, end, step) -> ttnn.Tensor:
    """Write `input_tensor` into the block `start..end` of `output_tensor`, in place; return `output_tensor`."""
    start, end, step = validate(input_tensor, output_tensor, start, end, step)
    spec, run_args, tensor_args = create_program_artifacts(input_tensor, output_tensor, start, end)
    # One dispatch. The destination is the program's output tensor: nothing is allocated (R3).
    ttnn.generic_op([input_tensor, output_tensor], spec, run_args, tensor_args)
    return output_tensor
