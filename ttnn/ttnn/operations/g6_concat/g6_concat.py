# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_concat: torch.cat(tensors, dim) on device tensors as one native device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
The program itself is described in g6_concat_program_descriptor.py.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_concat_program_descriptor as _program
from .g6_concat_program_descriptor import (
    K_MAX,
    L1_OP_BUDGET,
    build_program,
    refresh_addresses,
)


def _canonical_dim(dim, rank):
    return dim + rank if dim < 0 else dim


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads the scenario dict
#    {"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}
#    that eval/golden_tests/g6_concat/axes.py:scenario_of_call builds.
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs["dtype"]


def tag_layout(inputs, axes):
    return inputs["layout"]


def tag_concat_dim(inputs, axes):
    rank = len(inputs["shapes"][0])
    d = _canonical_dim(inputs["dim"], rank)
    if d == rank - 1:
        return "w"
    if d == rank - 2:
        return "h"
    return "outer"


def tag_alignment(inputs, axes):
    shapes = inputs["shapes"]
    if any(s[-1] % 32 for s in shapes):
        return "w_non_aligned"
    if any(s[-2] % 32 for s in shapes):
        return "h_non_aligned"
    return "tile_aligned"


def tag_input_buffer(inputs, axes):
    buffers = set(inputs["input_buffers"])
    if buffers == {"dram"}:
        return "dram"
    if buffers == {"l1"}:
        return "l1"
    return "mixed"


def tag_output_buffer(inputs, axes):
    return "l1" if inputs["memory_config"] == "l1" else "dram"


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "concat_dim": tag_concat_dim,
    "alignment": tag_alignment,
    "input_buffer": tag_input_buffer,
    "output_buffer": tag_output_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "concat_dim": ["outer", "h", "w"],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "input_buffer": ["dram"],
    "output_buffer": ["dram"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_name(buffer_type):
    return "l1" if buffer_type == ttnn.BufferType.L1 else "dram"


def _scenario(tensors, dim, memory_config):
    out_l1 = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
    return {
        "shapes": [list(t.shape) for t in tensors],
        "dim": dim,
        "dtype": tensors[0].dtype,
        "layout": tensors[0].layout,
        "input_buffers": [_buffer_name(t.memory_config().buffer_type) for t in tensors],
        "memory_config": "l1" if out_l1 else None,
    }


def _validate_structure(tensors, dim, memory_config):
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        raise ValueError("g6_concat: tensors must be a non-empty list of ttnn.Tensor")
    if not all(isinstance(t, ttnn.Tensor) for t in tensors):
        raise ValueError("g6_concat: every element of tensors must be a ttnn.Tensor")
    if len(tensors) > K_MAX:
        raise RuntimeError(f"g6_concat: {len(tensors)} inputs exceed K_MAX={K_MAX} (runtime-argument capacity)")
    if not isinstance(dim, int):
        raise ValueError(f"g6_concat: dim must be an int, got {type(dim).__name__}")

    shapes = [list(t.shape) for t in tensors]
    rank = len(shapes[0])
    if rank not in (3, 4):
        raise ValueError(f"g6_concat: rank {rank} not supported; inputs must be rank 3 or 4")
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"g6_concat: inputs differ in rank: {shapes}")
    if not -rank <= dim < rank:
        raise ValueError(f"g6_concat: dim={dim} outside [{-rank}, {rank - 1}] for rank {rank}")
    d = _canonical_dim(dim, rank)
    if any(t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError(f"g6_concat: inputs differ in dtype: {[t.dtype for t in tensors]}")
    if any(t.layout != tensors[0].layout for t in tensors):
        raise ValueError(f"g6_concat: inputs differ in layout: {[t.layout for t in tensors]}")
    for s in shapes:
        if any(s[i] != shapes[0][i] for i in range(rank) if i != d):
            raise ValueError(f"g6_concat: inputs differ outside dim {d}: {shapes}")
    for t in tensors:
        if t.storage_type() != ttnn.StorageType.DEVICE:
            raise ValueError("g6_concat: every input must be on device")
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("g6_concat: sharded inputs are not accepted; every input must be interleaved")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_concat: memory_config must be INTERLEAVED")


def validate(tensors, dim=0, *, memory_config=None):
    _validate_structure(tensors, dim, memory_config)

    scenario = _scenario(tensors, dim, memory_config)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(scenario, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_concat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_concat: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Program cache (BB-07): the descriptor is built once per key; a hit rewrites
# only the buffer-address common runtime args.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _output_memory_config(memory_config):
    return memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG


def _cache_key(tensors, d, out_mc, device):
    grid = device.compute_with_storage_grid_size()
    return (
        "g6_concat",
        tuple(tuple(t.shape) for t in tensors),
        d,
        tensors[0].dtype,
        tensors[0].layout,
        tuple(t.memory_config().buffer_type for t in tensors),
        out_mc.buffer_type,
        out_mc.memory_layout,
        (grid.x, grid.y),
        _program.MAX_CORES,
    )


def _output_shape(tensors, d):
    out_shape = list(tensors[0].shape)
    out_shape[d] = sum(t.shape[d] for t in tensors)
    return out_shape


def _program_for(tensors, d, output, key):
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        entry = build_program(list(tensors), output, d)
        entry.descriptor.custom_program_hash = hash(key) & 0xFFFFFFFFFFFFFFFF
        if entry.plan["l1_total_bytes"] > L1_OP_BUDGET:
            raise RuntimeError(
                f"g6_concat: per-core L1 footprint {entry.plan['l1_total_bytes']} B exceeds the "
                f"model's budget of {L1_OP_BUDGET} B"
            )
        _PROGRAM_CACHE[key] = entry
    return entry


def g6_concat_plan(tensors, dim=0, *, memory_config=None):
    """Report the plan a call selects (BB-08/BB-21) without dispatching anything."""
    validate(tensors, dim, memory_config=memory_config)
    d = _canonical_dim(dim, len(tensors[0].shape))
    out_mc = _output_memory_config(memory_config)
    device = tensors[0].device()
    key = _cache_key(tensors, d, out_mc, device)
    entry = _PROGRAM_CACHE.get(key)
    if entry is not None:
        return dict(entry.plan)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(_output_shape(tensors, d)), tensors[0].dtype, tensors[0].layout, device, out_mc
    )
    return dict(_program_for(tensors, d, output, key).plan)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_concat(tensors, dim=0, *, memory_config=None):
    validate(tensors, dim, memory_config=memory_config)

    d = _canonical_dim(dim, len(tensors[0].shape))
    out_mc = _output_memory_config(memory_config)
    device = tensors[0].device()

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(_output_shape(tensors, d)), tensors[0].dtype, tensors[0].layout, device, out_mc
    )

    key = _cache_key(tensors, d, out_mc, device)
    entry = _program_for(tensors, d, output, key)
    refresh_addresses(entry, list(tensors), output)

    return ttnn.generic_op([*tensors, output], entry.descriptor)
