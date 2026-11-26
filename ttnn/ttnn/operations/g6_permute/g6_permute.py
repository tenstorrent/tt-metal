# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute: torch.permute(x, dims) on a device tensor, as one device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate(). Every axis is projected
from a scenario dict {"input_shape", "dims", "dtype", "layout", "buffer"}; `inputs` is (scenario,).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_permute_program_descriptor import describe_plan_dict, get_plan, program_descriptor

_ROWS = (0, 2, 1, 3)


def canonical_dims(rank, dims):
    """dims with negative entries wrapped; ValueError (message names dims) unless a permutation of range(rank)."""
    dims = tuple(int(d) for d in dims)
    if len(dims) != rank:
        raise ValueError(f"g6_permute: dims {dims} has {len(dims)} entries, the tensor has rank {rank}")
    for d in dims:
        if d < -rank or d >= rank:
            raise ValueError(f"g6_permute: dims entry {d} of {dims} is outside [{-rank}, {rank - 1}]")
    wrapped = tuple(d + rank if d < 0 else d for d in dims)
    if sorted(wrapped) != list(range(rank)):
        raise ValueError(f"g6_permute: dims {dims} is not a permutation of range({rank})")
    return wrapped


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["input_shape"]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_dims(inputs, axes):
    scenario = inputs[0]
    return canonical_dims(len(scenario["input_shape"]), scenario["dims"])


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "buffer": tag_buffer,
    "rank": tag_rank,
    "alignment": tag_alignment,
    "dims": tag_dims,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "buffer": ["dram"],
    "rank": [4],
    "alignment": ["h_non_aligned"],
    "dims": [_ROWS],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}


def _scenario(input_tensor, dims):
    memory_config = input_tensor.memory_config()
    buffer = _BUFFER_NAME.get(memory_config.buffer_type, str(memory_config.buffer_type))
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        # No SUPPORTED list holds this value, so the per-axis check refuses it.
        buffer = f"{buffer}_{memory_config.memory_layout}"
    return {
        "input_shape": tuple(int(d) for d in input_tensor.shape),
        "dims": tuple(int(d) for d in dims),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "buffer": buffer,
    }


def validate(input_tensor, dims):
    """Structural dims checks (ValueError), then SUPPORTED per axis, then EXCLUSIONS. Returns canonical dims."""
    scenario = _scenario(input_tensor, dims)
    canonical = canonical_dims(len(scenario["input_shape"]), scenario["dims"])

    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_permute: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_permute: unsupported combination (refinement candidate): {exc}")

    return canonical


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor:
    """out = torch.permute(input_tensor, dims), in the input's dtype, layout and memory placement."""
    canonical = validate(input_tensor, dims)

    output_shape = [int(input_tensor.shape[d]) for d in canonical]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = program_descriptor(input_tensor, output_tensor, canonical)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def describe_plan(input_tensor: ttnn.Tensor, dims) -> dict:
    """The plan a g6_permute call with these arguments runs. Host only: no allocation, no dispatch."""
    canonical = validate(input_tensor, dims)
    return describe_plan_dict(get_plan(input_tensor, canonical))
