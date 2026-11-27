# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_permute: torch.permute(x, dims) as one device program (registry model).

Phase 0 builds regime rows_tile: TILE bfloat16 DRAM-interleaved rank 4, dims (0, 2, 1, 3).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_permute_program_descriptor import (
    READER_KERNEL_INDEX,
    WRITER_KERNEL_INDEX,
    Plan,
    build_plan,
    create_program_descriptor,
)

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}


def _canonical_dims(rank, dims):
    return tuple(int(d) + rank if int(d) < 0 else int(d) for d in dims)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,), scenario a dict with keys
#    input_shape, dims, dtype, layout, buffer.
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
    H, W = int(shape[-2]), int(shape[-1])
    if W % 32 != 0:
        return "w_non_aligned"
    if H % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_dims(inputs, axes):
    scenario = inputs[0]
    return _canonical_dims(len(scenario["input_shape"]), scenario["dims"])


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


def _check_dims_structure(rank, dims):
    try:
        entries = [int(d) for d in dims]
    except TypeError as exc:
        raise ValueError(f"g6_permute: dims must be a sequence of ints, got {dims!r}") from exc
    if len(entries) != rank:
        raise ValueError(f"g6_permute: dims {tuple(entries)} has {len(entries)} entries, the tensor has rank {rank}")
    for d in entries:
        if d < -rank or d >= rank:
            raise ValueError(f"g6_permute: dims entry {d} is outside [-{rank}, {rank - 1}]")
    if sorted(_canonical_dims(rank, entries)) != list(range(rank)):
        raise ValueError(f"g6_permute: dims {tuple(entries)} is not a permutation of range({rank})")


def _scenario(input_tensor, dims):
    buffer_type = input_tensor.memory_config().buffer_type
    return {
        "input_shape": tuple(int(d) for d in input_tensor.shape),
        "dims": tuple(int(d) for d in dims),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "buffer": _BUFFER_NAME.get(buffer_type, str(buffer_type)),
    }


def validate(input_tensor, dims):
    rank = len(input_tensor.shape)
    _check_dims_structure(rank, dims)

    memory_layout = input_tensor.memory_config().memory_layout
    if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue(f"g6_permute: memory_layout={memory_layout!r} is not INTERLEAVED")

    scenario = _scenario(input_tensor, dims)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_permute: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_permute: unsupported combination (refinement candidate): {exc}")

    return axes


# ---------------------------------------------------------------------------
# Plan report and descriptor cache
# ---------------------------------------------------------------------------

# One (plan, descriptor) per cache key; a hit only patches the two buffer addresses.
_PROGRAM_CACHE: dict = {}


def _cache_key(input_tensor, canonical_dims):
    mem = input_tensor.memory_config()
    grid = input_tensor.device().compute_with_storage_grid_size()
    return (
        tuple(int(d) for d in input_tensor.shape),
        tuple(canonical_dims),
        input_tensor.dtype,
        input_tensor.layout,
        mem.buffer_type,
        mem.memory_layout,
        (grid.x, grid.y),
    )


def _plan_for(input_tensor, canonical_dims) -> Plan:
    entry = _PROGRAM_CACHE.get(_cache_key(input_tensor, canonical_dims))
    if entry is not None:
        return entry[0]
    return build_plan(input_tensor, canonical_dims)


def g6_permute_plan(input_tensor: ttnn.Tensor, dims) -> Plan:
    """Report the plan (regime, core split, block knobs, per-core roles) a call selects."""
    axes = validate(input_tensor, dims)
    return _plan_for(input_tensor, axes["dims"])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_permute(input_tensor: ttnn.Tensor, dims) -> ttnn.Tensor:
    axes = validate(input_tensor, dims)
    canonical_dims = axes["dims"]

    out_shape = [int(input_tensor.shape[d]) for d in canonical_dims]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, canonical_dims)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        plan = build_plan(input_tensor, canonical_dims)
        descriptor = create_program_descriptor(plan, input_tensor, output_tensor)
        _PROGRAM_CACHE[key] = (plan, descriptor)
    else:
        _, descriptor = entry
        descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args = [input_tensor.buffer_address()]
        descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args = [output_tensor.buffer_address()]

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
