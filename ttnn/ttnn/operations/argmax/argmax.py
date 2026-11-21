# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""argmax along the last dimension, as one Metal 2.0 ProgramSpec dispatch.

Specification: eval/specs/argmax/spec.py. Design: op_design.md (a compare-and-index scan on the
data-movement harts, rows or row segments dealt to every scanner thread of the grid, SEGMENTED rows
merged by an in-program gather). Host program: argmax_program_artifacts.py.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .argmax_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, imported, never redefined.
# ---------------------------------------------------------------------------

from eval.specs.argmax.spec import AXES  # noqa: E402
from eval.memory import memory_axes  # noqa: E402

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding": ["interleaved"],
    "dim": [-1],
    "keepdim": [False],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_UINT32_LIMIT = 2**32


def _canonical_dim(dim, ndim):
    """R4: a negative offset, Torch convention; None stays None."""
    if dim is None:
        return None
    return dim if dim < 0 else dim - ndim


def validate(input_tensor, dim=None, keepdim=False, *, sub_core_grids=None, output_tensor=None):
    """Refuse malformed calls (X1, X2) and anything outside SUPPORTED / inside EXCLUSIONS (R10).

    `sub_core_grids` is never read here (R5): it says where the work runs, not what is computed.
    """
    # X2 / R6: the op allocates its result; it writes into no tensor of the caller's.
    if output_tensor is not None:
        raise ValueError("argmax: output_tensor must be None; the op allocates its own output")

    shape = tuple(input_tensor.shape)
    ndim = len(shape)
    # X1: dim outside the rank.
    if dim is not None:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(f"argmax: dim must be an int or None, got {type(dim).__name__}")
        if not (-ndim <= dim <= ndim - 1):
            raise IndexError(f"argmax: dim={dim} is out of range for a rank-{ndim} tensor [{-ndim}, {ndim - 1}]")

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory_axes(input_tensor),  # R9: memory_config and sharding
        "dim": _canonical_dim(dim, ndim),  # R4: canonicalised before the support check
        "keepdim": bool(keepdim),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"argmax: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"argmax: unsupported combination (refinement candidate): {exc}")

    # Mechanism caps: an index and a row count are carried as uint32.
    rows = math.prod(shape[:-1])
    if shape[-1] >= _UINT32_LIMIT or rows >= _UINT32_LIMIT:
        raise NotImplementedError("argmax: a row count or a row length of 2**32 or more does not fit a uint32 index")
    if shape[-1] == 0 or rows == 0:
        raise NotImplementedError("argmax: an empty input has no maximum")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_shape(shape, keepdim):
    """The input's shape with the last dimension removed, or kept with size 1."""
    return list(shape[:-1]) + [1] if keepdim else list(shape[:-1])


def argmax(
    input_tensor: ttnn.Tensor,
    dim: int = None,
    keepdim: bool = False,
    *,
    sub_core_grids: "ttnn.CoreRangeSet" = None,
    output_tensor: ttnn.Tensor = None,
) -> ttnn.Tensor:
    """Index of the largest element along `dim` (lowest index on ties, +0.0 == -0.0), as uint32 row-major."""
    return _argmax(input_tensor, dim, keepdim, sub_core_grids=sub_core_grids, output_tensor=output_tensor)


def _argmax(input_tensor, dim=None, keepdim=False, *, sub_core_grids=None, output_tensor=None, **probes):
    """The entry point with the test probes of create_program_artifacts (`probe_scanner_ids`)."""
    validate(input_tensor, dim, keepdim, sub_core_grids=sub_core_grids, output_tensor=output_tensor)

    device = input_tensor.device()
    out_shape = _output_shape(tuple(input_tensor.shape), keepdim)
    # R8: the output lives where the input does, interleaved.
    out_spec = ttnn.TensorSpec(
        ttnn.Shape(out_shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, memory_config=input_tensor.memory_config()
    )
    output = ttnn.allocate_tensor_on_device(out_spec, device)

    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, output, sub_core_grids=sub_core_grids, **probes
    )
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
