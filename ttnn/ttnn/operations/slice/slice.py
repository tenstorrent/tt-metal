# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""slice: the contiguous block input[s0:e0, ..., s(r-1):e(r-1)] of an interleaved tensor.

Registry-model op file (eval/op_template.py). Specification: eval/specs/slice/spec.py.
Design: op_design.md (regime `page_relay`: whole pages relayed through one DM->DM ring per core,
over the whole grid, in one ttnn.generic_op dispatch).

NOTE: the entry point is named `slice`, as the op is, so this module never uses Python's builtin
`slice`.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.slice.spec import AXES, canonical_step  # noqa: E402

from .slice_program_artifacts import DM_THREAD_PLAN, check_mechanism_caps, create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — the specification's PHASE0, widened by the refinements:
#    Refinement 1: sliced_region "w". Refinement 10: rank 4 (the 4-D page map already covers it; validate-only)
#    and step "ones" (a step of 1 in every dimension is the same block, R9; "other" is refused by name).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":           [ttnn.bfloat16],
    "layout":          [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config":   ["DRAM", "L1"],
    "sharding":        ["interleaved"],
    "sliced_region":   ["outer", "h", "w"],
    "start_alignment": ["tile_aligned"],
    "alignment":       ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":            [3, 4],
    "step":            [None, "ones"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _normalize_bounds(input_tensor, slice_start, slice_end):
    """(shape, start, end) with negative indices counted from the end (R2), after the malformed-call
    checks (X1-X3). Accepts lists or tuples (R3)."""
    shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(shape)
    for name, bounds in (("slice_start", slice_start), ("slice_end", slice_end)):
        if not isinstance(bounds, (list, tuple)):
            raise ValueError(f"slice: {name} must be a list or a tuple, got {type(bounds).__name__}")
        if len(bounds) != rank:
            raise ValueError(f"slice: {name} has {len(bounds)} entries for a tensor of rank {rank}")
        for i in bounds:
            if isinstance(i, bool) or not isinstance(i, int):
                raise ValueError(f"slice: {name} entries must be ints, got {i!r}")
    start = tuple(s + size if s < 0 else s for s, size in zip(slice_start, shape))
    end = tuple(e + size if e < 0 else e for e, size in zip(slice_end, shape))
    for d, (s, e, size) in enumerate(zip(start, end, shape)):
        if e > size or e < 0:
            raise ValueError(f"slice: end {slice_end[d]} is beyond dimension {d} of size {size}")
        if s < 0 or s > size:
            raise ValueError(f"slice: start {slice_start[d]} is outside dimension {d} of size {size}")
        if e < s:
            raise ValueError(f"slice: end {slice_end[d]} is before start {slice_start[d]} in dimension {d}")
    for d, (s, e) in enumerate(zip(start, end)):
        if e == s:
            raise NotImplementedError(f"slice: an empty block (dimension {d}: {s}:{e}) is not supported")
    return shape, start, end


def _axes(input_tensor, shape, start, end, slice_step):
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout, **memory_axes(input_tensor),
            # the `step` axis: the specification's own canonicalization (None / "ones" / "other", R9)
            "step": canonical_step(slice_step, None)}
    inputs = (shape, start, end)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(input_tensor, slice_start, slice_end, slice_step=None):
    """Malformed calls raise ValueError (REJECTIONS); valid calls outside support raise the typed
    NotImplementedError refusals (R6). Returns the normalized (shape, start, end)."""
    shape, start, end = _normalize_bounds(input_tensor, slice_start, slice_end)
    if slice_step is not None and (not isinstance(slice_step, (list, tuple)) or len(slice_step) != len(shape)):
        raise ValueError("slice: slice_step must be a list or a tuple with one entry per dimension")
    axes = _axes(input_tensor, shape, start, end, slice_step)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"slice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"slice: unsupported combination (refinement candidate): {exc}")

    check_mechanism_caps(input_tensor)
    return shape, start, end


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _dispatch(input_tensor, start, end, threads):
    """Allocate the output in the input's buffer (R7) and run the one program (page_relay)."""
    out_shape = [e - s for s, e in zip(start, end)]
    output = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape(out_shape),
            input_tensor.dtype,
            input_tensor.layout,
            buffer_type=input_tensor.memory_config().buffer_type,  # R7: the input's buffer, interleaved
        ),
        input_tensor.device(),
    )
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, start, threads=threads)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)


def _slice(input_tensor, slice_start, slice_end, *, threads=DM_THREAD_PLAN):
    """The op with its thread plan exposed: tests turn it, the public entry point uses the default."""
    _, start, end = validate(input_tensor, slice_start, slice_end)
    return _dispatch(input_tensor, start, end, threads)


def slice(  # noqa: A001  (the op's name; shadows the builtin in this module only)
    input_tensor: ttnn.Tensor,
    slice_start,
    slice_end,
    slice_step=None,
) -> ttnn.Tensor:
    _, start, end = validate(input_tensor, slice_start, slice_end, slice_step)
    return _dispatch(input_tensor, start, end, DM_THREAD_PLAN)
