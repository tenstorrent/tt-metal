# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""permute: torch.permute for a TTNN tensor, one ProgramSpec dispatch per call (Quasar, Metal 2.0).

Specification: eval/specs/permute/spec.py. Design: op_design.md beside this file.

The host contract, in order (op_design.md "Host-side contract"):
  1. dims -> list (R3)                       4. axes from the specification's AXES + memory_axes (R8)
  2. reject malformed dims (X1, X2, X3)      5. validate() against SUPPORTED / EXCLUSIONS (R6)
  3. negative entries -> d + rank (R2)       6. allocate the output in the input's buffer (R7), dispatch one program
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from eval.specs.permute.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .permute_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding": ["interleaved"],
    "permutation": ["leading", "last_two", "general"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [3, 4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none in the first phase.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# dims handling (REJECTIONS X1-X3, R2, R3)
# ---------------------------------------------------------------------------


def normalize_dims(dims, rank):
    """`dims` as a list of indices in [0, rank): negative entries count from the last dimension (R2).

    Raises ValueError for a malformed permutation: wrong length (X1), an entry outside [-rank, rank-1] (X3),
    a dimension named twice (X2)."""
    try:
        dims = [int(x) for x in dims]  # R3: a list or a tuple
    except TypeError as e:
        raise ValueError(f"permute: dims must be a sequence of ints, got {dims!r}") from e
    if len(dims) != rank:
        raise ValueError(f"permute: dims {dims} has {len(dims)} entries for a tensor of rank {rank}")
    for x in dims:
        if not -rank <= x < rank:
            raise ValueError(f"permute: dims entry {x} is outside [{-rank}, {rank - 1}]")
    norm = [x + rank if x < 0 else x for x in dims]
    if sorted(norm) != list(range(rank)):
        raise ValueError(f"permute: dims {dims} names a dimension more than once")
    return norm


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, dims):
    shape = tuple(int(x) for x in input_tensor.shape)
    norm = normalize_dims(dims, len(shape))
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
    axes.update(memory_axes(input_tensor))  # R8: memory_config + sharding
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger((shape, tuple(norm)), axes)
    return axes, norm


def validate(input_tensor, dims):
    """Raise NotImplementedError (UnsupportedAxisValue / ExcludedCell) for any call outside SUPPORTED (R6),
    ValueError for a malformed `dims` (REJECTIONS)."""
    axes, norm = _axes(input_tensor, dims)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"permute: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"permute: unsupported combination (refinement candidate): {exc}")
    return norm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_spec(input_tensor, norm):
    shape = [int(input_tensor.shape[d]) for d in norm]
    # R7: interleaved, in the input's buffer type.
    return ttnn.TensorSpec(
        ttnn.Shape(shape), input_tensor.dtype, input_tensor.layout, buffer_type=input_tensor.memory_config().buffer_type
    )


def _permute_into(input_tensor, dims, output_tensor, **knobs):
    """Run the permute program writing into a caller-given output (tests pre-fill it with a sentinel to see
    that every thread of the plan did its share). `dims` must already be validated and normalized; `knobs`
    turn the regime's block knobs and thread plan (permute_program_artifacts.BUILDERS)."""
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output_tensor, dims, **knobs)
    return ttnn.generic_op([input_tensor, output_tensor], spec, run_args, tensor_indices)


def permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor:
    norm = validate(input_tensor, dims)
    output = ttnn.allocate_tensor_on_device(_output_spec(input_tensor, norm), input_tensor.device())
    return _permute_into(input_tensor, norm, output)
