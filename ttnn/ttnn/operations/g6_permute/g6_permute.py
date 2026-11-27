# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute: torch.permute(input_tensor, dims) as one device program.

Phase 0 builds regime R1 `rows_tile` (op_design.md): dims (0, 2, 1, 3) on a bfloat16 TILE
tensor interleaved in DRAM, every value moved bit for bit.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_permute_program_descriptor import (
    build_plan,
    create_program_descriptor,
    patch_addresses,
    plan_key,
    public_plan,
)


def _wrap_dims(rank, dims):
    return tuple(int(d) + rank if int(d) < 0 else int(d) for d in dims)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), scenario = {"input_shape", "dims", "dtype", "layout", "buffer"}
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
    return _wrap_dims(len(scenario["input_shape"]), scenario["dims"])


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
    "dims": [(0, 2, 1, 3)],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}


def _buffer_axis(input_tensor):
    memory_config = input_tensor.memory_config()
    name = _BUFFER_NAME.get(memory_config.buffer_type, str(memory_config.buffer_type))
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        # No SUPPORTED list holds a sharded placement, so the per-axis check refuses it.
        return f"{name}_{memory_config.memory_layout}"
    return name


def _check_dims(rank, dims):
    """Structural checks; tag_dims needs a well-formed dims, so these run before SUPPORTED."""
    dims = tuple(int(d) for d in dims)
    if len(dims) != rank:
        raise ValueError(f"g6_permute: dims {dims} has {len(dims)} entries, the input has rank {rank}")
    for d in dims:
        if d < -rank or d >= rank:
            raise ValueError(f"g6_permute: dims entry {d} is outside [-{rank}, {rank - 1}]")
    if sorted(_wrap_dims(rank, dims)) != list(range(rank)):
        raise ValueError(f"g6_permute: dims {dims} is not a permutation of range({rank})")
    return dims


def validate(input_tensor, dims):
    """Returns the canonical dims of a supported call; raises for anything else."""
    rank = len(input_tensor.shape)
    dims = _check_dims(rank, dims)
    scenario = {
        "input_shape": tuple(int(d) for d in input_tensor.shape),
        "dims": dims,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "buffer": _buffer_axis(input_tensor),
    }
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_permute: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exclusion in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exclusion.items()):
            raise ExcludedCell(f"g6_permute: unsupported combination (refinement candidate): {exclusion}")

    return axes["dims"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Keyed by plan_key(): the plan and the descriptor are built once per signature; a repeat call
# patches the two buffer addresses and dispatches.
_PLANS = {}
_PROGRAMS = {}


def _cached_plan(input_tensor, canonical_dims):
    key = plan_key(input_tensor, canonical_dims)
    plan = _PLANS.get(key)
    if plan is None:
        plan = build_plan(input_tensor)
        _PLANS[key] = plan
    return key, plan


def describe_plan(input_tensor, dims):
    """The plan a call with these arguments runs (host only: no allocation, no dispatch)."""
    canonical_dims = validate(input_tensor, dims)
    _, plan = _cached_plan(input_tensor, canonical_dims)
    return public_plan(plan)


def g6_permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor:
    canonical_dims = validate(input_tensor, dims)
    key, plan = _cached_plan(input_tensor, canonical_dims)

    output_shape = [int(input_tensor.shape[d]) for d in canonical_dims]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    descriptor = _PROGRAMS.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(input_tensor, output_tensor, plan, key)
        _PROGRAMS[key] = descriptor
    else:
        patch_addresses(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
