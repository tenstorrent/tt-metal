# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat: torch.cat(tensors, dim) on device tensors in one program, bit for bit.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() gate the
call; the entry point dispatches exactly one generic_op.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_concat_program_descriptor as program_descriptor
from .g6_concat_program_descriptor import K_MAX, TILE_DIM, build_plan, build_program, canonical_dim


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is the scenario dict of eval/golden_tests/g6_concat/axes.py:
#    {"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}.
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs["dtype"]


def tag_layout(inputs, axes):
    return inputs["layout"]


def tag_concat_dim(inputs, axes):
    rank = len(inputs["shapes"][0])
    d = canonical_dim(inputs["dim"], rank)
    if d == rank - 1:
        return "w"
    if d == rank - 2:
        return "h"
    return "outer"


def tag_alignment(inputs, axes):
    shapes = inputs["shapes"]
    if any(s[-1] % TILE_DIM != 0 for s in shapes):
        return "w_non_aligned"
    if any(s[-2] % TILE_DIM != 0 for s in shapes):
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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_name(buffer_type):
    return "l1" if buffer_type == ttnn.BufferType.L1 else "dram"


def scenario_of_call(tensors, dim=0, *, memory_config=None):
    """The golden scenario dict of a call (same construction as axes.py:scenario_of_call)."""
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
        raise ValueError("g6_concat: tensors must be a non-empty list of device tensors")
    for t in tensors:
        if not isinstance(t, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(t):
            raise ValueError("g6_concat: every input must be a ttnn.Tensor on device")
    shapes = [list(t.shape) for t in tensors]
    rank = len(shapes[0])
    if rank not in (3, 4):
        raise ValueError(f"g6_concat: rank must be 3 or 4, got {rank}")
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"g6_concat: inputs differ in rank: {shapes}")
    if any(t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError("g6_concat: inputs differ in dtype")
    if any(t.layout != tensors[0].layout for t in tensors):
        raise ValueError("g6_concat: inputs differ in layout")
    if not isinstance(dim, int) or not -rank <= dim < rank:
        raise ValueError(f"g6_concat: dim {dim} outside [{-rank}, {rank - 1}]")
    d = canonical_dim(dim, rank)
    for s in shapes:
        if any(s[i] != shapes[0][i] for i in range(rank) if i != d):
            raise ValueError(f"g6_concat: sizes differ outside dim {d}: {shapes}")
    if any(size == 0 for s in shapes for size in s):
        raise ValueError(f"g6_concat: zero-size inputs are not accepted: {shapes}")
    for t in tensors:
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("g6_concat: sharded inputs are refused; every input must be interleaved")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_concat: memory_config must be INTERLEAVED")
    if len(tensors) > K_MAX:
        raise RuntimeError(f"g6_concat: {len(tensors)} inputs exceed K_MAX = {K_MAX}")


def validate(tensors, dim=0, *, memory_config=None):
    _validate_structure(tensors, dim, memory_config)

    scenario = scenario_of_call(tensors, dim, memory_config=memory_config)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(scenario, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_concat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_concat: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Program cache (BB-07): one plan + descriptor per key; a hit only rewrites buffer addresses.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _cache_key(tensors, d, output_memory_config, device):
    grid = device.compute_with_storage_grid_size()
    return (
        "g6_concat",
        device.id(),
        tuple(tuple(t.shape) for t in tensors),
        d,
        str(tensors[0].dtype),
        str(tensors[0].layout),
        tuple(str(t.memory_config().buffer_type) for t in tensors),
        str(output_memory_config.buffer_type),
        str(output_memory_config.memory_layout),
        (grid.x, grid.y),
        program_descriptor.blocking_knobs(),
    )


def _output_memory_config(memory_config):
    return memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG


def _cached_plan(tensors, d, output_memory_config, device):
    key = _cache_key(tensors, d, output_memory_config, device)
    entry = _PROGRAM_CACHE.get(key)
    plan = entry.plan if entry is not None else build_plan(tensors, d, output_memory_config, device)
    return key, entry, plan


def g6_concat_plan(tensors, dim=0, *, memory_config=None):
    """The plan a call selects (regime, cores, block, CB bytes, per-core role); dispatches nothing."""
    validate(tensors, dim, memory_config=memory_config)
    d = canonical_dim(dim, len(tensors[0].shape))
    _, _, plan = _cached_plan(list(tensors), d, _output_memory_config(memory_config), tensors[0].device())
    return plan.report()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_concat(tensors, dim=0, *, memory_config=None):
    validate(tensors, dim, memory_config=memory_config)

    tensors = list(tensors)
    d = canonical_dim(dim, len(tensors[0].shape))
    device = tensors[0].device()
    output_memory_config = _output_memory_config(memory_config)

    key, entry, plan = _cached_plan(tensors, d, output_memory_config, device)

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.out_shape),
        tensors[0].dtype,
        tensors[0].layout,
        device,
        output_memory_config,
    )

    if entry is None:
        entry = build_program(plan, tensors, output, hash(key) & 0xFFFFFFFFFFFFFFFF)
        _PROGRAM_CACHE[key] = entry
    descriptor = entry.bind(tensors, output)

    return ttnn.generic_op([*tensors, output], descriptor)
