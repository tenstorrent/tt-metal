# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""plus_one: every element of an int32 / uint32 row-major tensor incremented in place, wrapping.

Specification: eval/specs/plus_one/spec.py. Design: op_design.md (same directory). One `ttnn.generic_op`
dispatch of a Metal 2.0 ProgramSpec whose only kernel is a multi-threaded data-movement worker
(plus_one_program_artifacts.py, kernels/plus_one_worker.cpp).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.plus_one.spec import AXES

from .plus_one_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — the first phase: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":                 [ttnn.int32, ttnn.uint32],
    "layout":                [ttnn.ROW_MAJOR_LAYOUT],
    "memory_config":         ["DRAM"],
    "sharding":              ["interleaved"],
    "skip_negative_entries": [False, True],
    "rank":                  [1],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, skip_negative_entries):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory.memory_axes(input_tensor),
        "skip_negative_entries": bool(skip_negative_entries),
    }
    inputs = (tuple(int(d) for d in input_tensor.shape),)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(input_tensor, *, skip_negative_entries=False):
    axes = _axes(input_tensor, skip_negative_entries)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"plus_one: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"plus_one: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _plus_one(input_tensor, *, skip_negative_entries=False, **knobs):
    """The dispatch, with the program builder's knobs and probes (`worker_threads`, `min_worker_bytes`,
    `scratch_depth`, `probe_tag_threads`) open to tests. Not part of the public signature."""
    validate(input_tensor, skip_negative_entries=skip_negative_entries)
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, skip_negative_entries=skip_negative_entries, **knobs
    )
    # One-element io_tensors: the tensor is the program's input and its output (rule R3).
    ttnn.generic_op([input_tensor], spec, run_args, tensor_indices)
    return input_tensor


def plus_one(
    input_tensor: ttnn.Tensor,
    *,
    skip_negative_entries: bool = False,
) -> ttnn.Tensor:
    """Increment every element of `input_tensor` by one, in its own buffer, and return that tensor.

    With `skip_negative_entries=True` only elements whose 32 bits, read as int32, are in [0, INT32_MAX) are
    incremented. 32-bit arithmetic wraps.
    """
    return _plus_one(input_tensor, skip_negative_entries=skip_negative_entries)
