# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""split: cut a tensor into consecutive chunks along one dimension (torch.split with an integer size).

Registry-model op file (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate(), and the
public entry point. The specification is eval/specs/split/spec.py; the design is op_design.md beside this file.

One `ttnn.generic_op` dispatch per call writes every chunk (Metal 2.0 ProgramSpec, data-movement kernels only:
the op moves values and computes nothing).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.split.spec import AXES  # noqa: E402

from .split_program_artifacts import create_program_artifacts

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
    "dim": [-2],
    "chunks": [2],
    "last_chunk": ["full"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [3],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none inside the Phase 0 rectangle.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_call(input_tensor, split_size, dim):
    """The malformed calls of the specification's REJECTIONS (X1-X3). Returns the canonical (negative) dim."""
    # X3: tt-metal's second form (a list of chunk sizes) does not exist here. bool is an int subclass: refuse it.
    if isinstance(split_size, bool) or not isinstance(split_size, int):
        raise TypeError(f"split: split_size must be an int, got {type(split_size).__name__} ({split_size!r})")
    # X1
    if split_size <= 0:
        raise ValueError(f"split: split_size must be greater than 0, got {split_size}")
    rank = len(input_tensor.shape)
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(f"split: dim must be an int, got {type(dim).__name__}")
    # X2
    if not -rank <= dim < rank:
        raise IndexError(f"split: dim {dim} is outside the tensor's rank {rank} (valid: [{-rank}, {rank - 1}])")
    # R3: canonicalize to a negative offset before the support check.
    return dim if dim < 0 else dim - rank


def _axes(input_tensor, split_size, dim):
    shape = tuple(int(d) for d in input_tensor.shape)
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout, **memory_axes(input_tensor)}
    # The taggers take the call's (shape, split_size, dim); tag_dim canonicalizes dim itself (R3).
    inputs = (shape, split_size, dim)
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)
    return axes


def validate(input_tensor, split_size, dim=0):
    """Raise for a malformed call (REJECTIONS) and refuse anything outside SUPPORTED / inside EXCLUSIONS (R7)."""
    _check_call(input_tensor, split_size, dim)
    axes = _axes(input_tensor, split_size, dim)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"split: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"split: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _chunk_memory_config(axes):
    # R5: every chunk in the input's buffer, interleaved, as tt-metal's op does with no memory config given.
    return ttnn.L1_MEMORY_CONFIG if axes["memory_config"] == "L1" else ttnn.DRAM_MEMORY_CONFIG


def split(
    input_tensor: ttnn.Tensor,
    split_size: int,
    dim: int = 0,
) -> list:
    """Cut `input_tensor` along `dim` into ceil(shape[dim] / split_size) chunks; returns a list of tensors."""
    axes = validate(input_tensor, split_size, dim)
    # axes["dim"] is the canonical negative dim (tag_dim, R3); validate() already checked the call.
    return _dispatch(input_tensor, split_size, axes["dim"], _chunk_memory_config(axes))


def _dispatch(input_tensor, split_size, d, memory_config):
    """Allocate every chunk and run the one program that writes them. `d` is the canonical (negative) dim.
    Below validate(): the tests call it directly to exercise chunk counts the kernels handle before SUPPORTED
    claims them."""
    shape = [int(s) for s in input_tensor.shape]
    extent = shape[d]
    device = input_tensor.device()

    chunks = []
    for start in range(0, extent, split_size):
        chunk_shape = list(shape)
        chunk_shape[d] = min(split_size, extent - start)
        spec = ttnn.TensorSpec(
            ttnn.Shape(chunk_shape), input_tensor.dtype, input_tensor.layout, memory_config=memory_config
        )
        chunks.append(ttnn.allocate_tensor_on_device(spec, device))

    program_spec, run_args, tensor_args = create_program_artifacts(input_tensor, chunks, split_size, d)
    # generic_op returns the last io tensor; the result is the whole list of chunks.
    ttnn.generic_op([input_tensor, *chunks], program_spec, run_args, tensor_args)
    return chunks
