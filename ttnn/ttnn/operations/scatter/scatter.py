# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scatter: `torch.scatter(input, dim, index, src)` as one Metal 2.0 ProgramSpec dispatch.

Registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate) and the public entry point. The
specification is eval/specs/scatter/spec.py; the design op_design.md beside this file.
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from eval.specs.scatter.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scatter_program_artifacts import create_program_artifacts, resident_line_fits

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: equals the specification's PHASE0 after the first phase.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding":      ["interleaved"],
    "dim":           [-2],
    "index_form":    ["lines"],
    "alignment":     ["tile_aligned", "h_non_aligned"],
    "rank":          [2],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _entry(input, dim, index, src):
    """The specification's INPUTS-entry form of a call: (shape of input, dim, shape of index, shape of src)."""
    return (tuple(input.shape), dim, tuple(index.shape), tuple(src.shape))


def validate(input, dim, index, src):
    """Refuse malformed calls (REJECTIONS) and every call outside SUPPORTED / inside EXCLUSIONS.

    Returns the canonical (negative) `dim`.
    """
    rank = len(input.shape)
    # X2: index and src must have the rank of input.
    if len(index.shape) != rank or len(src.shape) != rank:
        raise ValueError(
            f"scatter: index (rank {len(index.shape)}) and src (rank {len(src.shape)}) must have the rank of "
            f"input ({rank})"
        )
    # X1: dim must be a dimension of the tensor.
    if not -rank <= dim < rank:
        raise IndexError(f"scatter: dim={dim} is not a dimension of a rank-{rank} tensor")
    # R4: canonicalize dim to its negative offset before the support check.
    dim = dim if dim < 0 else dim - rank

    entry = _entry(input, dim, index, src)
    axes = {
        "dtype": input.dtype,
        "layout": input.layout,
    }
    # R8: placement axes of each of the three tensors; they must agree.
    placements = [memory_axes(t) for t in (input, index, src)]
    if any(p != placements[0] for p in placements[1:]):
        raise UnsupportedAxisValue(
            f"scatter: input, index and src must live in one place, got {placements}"
        )
    axes.update(placements[0])
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(entry, axes)

    # R9 (X3, X4): `index_form` "other" refused by name.
    if axes["index_form"] != "lines":
        raise UnsupportedAxisValue(
            f"scatter: index_form={axes['index_form']!r} (index {tuple(index.shape)}, src {tuple(src.shape)}, "
            f"input {tuple(input.shape)}, dim {dim}) not in SUPPORTED {SUPPORTED['index_form']}"
        )
    # R5: INT32 row-major index; src of input's dtype and layout.
    if index.dtype != ttnn.int32 or index.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise UnsupportedAxisValue(
            f"scatter: index must be INT32 ROW_MAJOR, got {index.dtype} {index.layout}"
        )
    if src.dtype != input.dtype or src.layout != input.layout:
        raise UnsupportedAxisValue(
            f"scatter: src ({src.dtype}, {src.layout}) must have input's dtype and layout "
            f"({input.dtype}, {input.layout})"
        )

    # R10: SUPPORTED per axis, then EXCLUSIONS.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"scatter: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"scatter: unsupported combination (refinement candidate): {exc}")

    # Regime: the whole output line must be resident (resident_line_block); banded_output_line is deferred.
    if not resident_line_fits(input, index):
        raise UnsupportedAxisValue(
            "scatter: the output line does not fit L1 (regime banded_output_line is not built)"
        )
    return dim


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scatter(
    input: ttnn.Tensor,
    dim: int,
    index: ttnn.Tensor,
    src: ttnn.Tensor,
) -> ttnn.Tensor:
    """A copy of `input` with `src`'s elements written at the positions `index` names along `dim`."""
    dim = validate(input, dim, index, src)
    # R7 / R3: a new tensor with input's shape, dtype, layout and memory config.
    output = ttnn.allocate_tensor_on_device(input.spec, input.device())
    spec, run_args, tensor_indices = create_program_artifacts(input, index, src, output, dim=dim)
    return ttnn.generic_op([input, index, src, output], spec, run_args, tensor_indices)
