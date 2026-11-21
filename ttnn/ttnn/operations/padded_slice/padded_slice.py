# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""padded_slice: a rank-4 block `input[s0:e0, s1:e1, s2:e2, s3:e3]`, flattened to rows
`[1, 1, n*h*w, width]` and written straight into the caller's height-sharded L1 `memory_config`.

Specification: eval/specs/padded_slice/spec.py. Design: op_design.md (regime A: each shard core
gathers its own rows from DRAM into its own shard, one NoC read per row, one dispatch).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.padded_slice import spec as _spec
from eval.specs.padded_slice.spec import AXES

from .padded_slice_program_artifacts import create_program_artifacts, GATHER_THREADS

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — the first phase: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding": ["interleaved"],
    "to": ["L1/height"],
    "step": ["ones"],
    "sliced_region": ["rows"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Malformed calls (REJECTIONS X1-X3): checked before any axis is derived.
# ---------------------------------------------------------------------------


def _bounds(input_tensor, padded_slice_start, padded_slice_end):
    """`start`, `end` as tuples of ints (lists accepted, R8), checked against the tensor."""
    shape = tuple(int(d) for d in input_tensor.shape)
    try:
        start = tuple(int(i) for i in padded_slice_start)
        end = tuple(int(i) for i in padded_slice_end)
    except TypeError as error:
        raise TypeError(f"padded_slice: start and end must be sequences of ints ({error})") from None
    if len(start) != len(shape) or len(end) != len(shape):
        raise ValueError(
            f"padded_slice: start {list(start)} and end {list(end)} need one entry per dimension of the "
            f"rank-{len(shape)} tensor {list(shape)}"
        )
    for d, (s, e, size) in enumerate(zip(start, end, shape)):
        if s < 0 or e < 0:
            raise ValueError(f"padded_slice: negative index in dimension {d} (start {s}, end {e})")
        if e > size:
            raise ValueError(f"padded_slice: end {e} of dimension {d} is beyond its size {size}")
        if e < s:
            raise ValueError(f"padded_slice: end {e} of dimension {d} is before its start {s}")
    if any(e == s for s, e in zip(start, end)):
        raise NotImplementedError(f"padded_slice: an empty block ({list(start)} to {list(end)}) is not specified")
    return shape, start, end


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, padded_slice_start, padded_slice_end, padded_slice_step, memory_config):
    shape, start, end = _bounds(input_tensor, padded_slice_start, padded_slice_end)
    arguments = {"padded_slice_start": start, "padded_slice_end": end}
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory.memory_axes(input_tensor),  # memory_config, sharding of the input (R9)
        # canonical_to raises TypeError for a non-config (X5), ValueError for narrow shards (X6).
        "to": _spec.canonical_to(memory_config, arguments),
        "step": _spec.canonical_step(padded_slice_step, arguments),
    }
    inputs = (shape, start, end)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes, shape, start, end


def validate(input_tensor, padded_slice_start, padded_slice_end, padded_slice_step=None, *, memory_config):
    """The runtime gate: malformed calls raise (X1-X6); calls outside SUPPORTED/EXCLUSIONS raise
    NotImplementedError (UnsupportedAxisValue / ExcludedCell). Returns the axes and the bounds."""
    axes, shape, start, end = _axes(input_tensor, padded_slice_start, padded_slice_end, padded_slice_step,
                                    memory_config)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"padded_slice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"padded_slice: unsupported combination (refinement candidate): {exc}")
    return axes, shape, start, end


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def padded_slice(
    input_tensor: ttnn.Tensor,
    padded_slice_start,
    padded_slice_end,
    padded_slice_step=None,
    *,
    memory_config: "ttnn.MemoryConfig",
) -> ttnn.Tensor:
    _, _, start, end = validate(input_tensor, padded_slice_start, padded_slice_end, padded_slice_step,
                                memory_config=memory_config)
    return _dispatch(input_tensor, start, end, memory_config)


def _dispatch(input_tensor, start, end, memory_config, *, gather_threads=GATHER_THREADS, max_block_rows=None):
    """Allocate the result in the caller's config (R5) and run the one gather program.

    `gather_threads` and `max_block_rows` are the design's knobs (thread plan R, block extent); the public
    entry point uses their host defaults, tests turn them."""
    n, h, w, _ = (e - s for s, e in zip(start, end))
    width = int(memory_config.shard_spec.shape[1])
    output_shape = ttnn.Shape([1, 1, n * h * w, width])
    output_spec = ttnn.TensorSpec(output_shape, input_tensor.dtype, ttnn.ROW_MAJOR_LAYOUT, memory_config)
    output = ttnn.allocate_tensor_on_device(output_spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, output, start, end, gather_threads=gather_threads, max_block_rows=max_block_rows
    )
    # The output is the last io tensor: generic_op returns it.
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
