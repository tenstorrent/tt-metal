# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""concat: `torch.cat(tensors, dim)` as one Metal 2.0 `ttnn.ProgramSpec` dispatch.

The op moves values and computes nothing: a reader gathers each output tile from the input pages (whole pages
when every seam sits on a tile boundary, byte runs otherwise) into one ring, and a writer stores the ring's
entries to the output. The output is always DRAM interleaved (spec R5). Elements of the output's tile padding
are not written; their values are undefined (they are not part of the output's data).

Registry model: INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS, validate(). The specification
is eval/specs/concat/spec.py.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.concat.spec import AXES  # noqa: E402

from .concat_program_artifacts import BLOCK_TILES, create_program_artifacts, select_regime


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES


# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding":      ["interleaved"],
    "dim":           [-1, -2],
    "count":         [2, 3],
    "alignment":     ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":          [3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_well_formed(tensors, dim):
    """The malformed calls of the specification's REJECTIONS (X1..X4), before any support check."""
    if tensors is None or len(tensors) == 0:
        raise ValueError("concat: the list of tensors is empty")  # X4
    shapes = [tuple(t.shape) for t in tensors]
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"concat: the tensors differ in rank: {shapes}")  # X2
    if not -rank <= dim <= rank - 1:
        raise IndexError(f"concat: dim={dim} is outside [-{rank}, {rank - 1}]")  # X3
    canonical = dim if dim < 0 else dim - rank
    for s in shapes[1:]:
        for d in range(-rank, 0):
            if d != canonical and s[d] != shapes[0][d]:
                raise ValueError(f"concat: the tensors differ in dimension {d}, which is not dim: {shapes}")  # X1
    return shapes


def classify(tensors, dim=0):
    """The axes of a call: tensor properties, memory axes of every tensor (R6), and every tagger."""
    shapes = _check_well_formed(tensors, dim)
    first = tensors[0]
    if any(t.dtype != first.dtype for t in tensors) or any(t.layout != first.layout for t in tensors):
        raise NotImplementedError("concat: tensors of one call must share dtype and layout")
    placements = [memory_axes(t) for t in tensors]
    if any(p != placements[0] for p in placements):
        raise NotImplementedError(f"concat: tensors of one call must share one placement, got {placements}")
    if any(t.device() != first.device() for t in tensors[1:]):
        raise NotImplementedError("concat: tensors of one call must be on one device")
    axes = {"dtype": first.dtype, "layout": first.layout, **placements[0]}
    inputs = (dim, *shapes)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)  # "dim" is canonicalised by its tagger (R2)
    return axes


def validate(tensors, dim=0):
    axes = classify(tensors, dim)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"concat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"concat: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _dispatch(tensors, dim, *, block_tiles=BLOCK_TILES):
    """Allocate the output and run the one program. `dim` is canonical (negative)."""
    first = tensors[0]
    out_shape = list(first.shape)
    out_shape[dim] = sum(int(t.shape[dim]) for t in tensors)
    # The output has the tensors' dtype and layout (spec OUTPUT) and is DRAM interleaved whatever their buffer (R5).
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), first.dtype, first.layout, first.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    spec, run_args, tensor_indices = create_program_artifacts(list(tensors), output, dim, block_tiles=block_tiles)
    return ttnn.generic_op([*tensors, output], spec, run_args, tensor_indices)


def concat_with_block_tiles(tensors, dim=0, *, block_tiles=BLOCK_TILES):
    """The op with its block knob turned (tests only; the public entry point keeps BLOCK_TILES)."""
    axes = validate(tensors, dim)
    return _dispatch(tensors, axes["dim"], block_tiles=block_tiles)


def concat(
    tensors: list,
    dim: int = 0,
) -> ttnn.Tensor:
    """`torch.cat(tensors, dim)`: one device-program dispatch, output DRAM interleaved."""
    axes = validate(tensors, dim)
    return _dispatch(tensors, axes["dim"])
