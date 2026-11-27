# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_concat: join K interleaved device tensors along one dim, bit for bit, in one program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
Every tagger reads the scenario dict
{"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}
that validate() builds from the call, exactly as the golden harness builds it.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_concat_program_descriptor import K_MAX, built_program, create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _canonical_dim(dim, rank):
    return dim % rank


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
    if any(int(s[-1]) % 32 for s in shapes):
        return "w_non_aligned"
    if any(int(s[-2]) % 32 for s in shapes):
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


def _check_structure(tensors, dim, memory_config):
    """Structural errors (not support refusals): ValueError."""
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        raise ValueError("g6_concat: tensors must be a non-empty list")
    if len(tensors) > K_MAX:
        raise ValueError(f"g6_concat: at most {K_MAX} inputs, got {len(tensors)}")
    for k, t in enumerate(tensors):
        if not isinstance(t, ttnn.Tensor):
            raise ValueError(f"g6_concat: input {k} is not a ttnn.Tensor")
        if not ttnn.is_tensor_storage_on_device(t):
            raise ValueError(f"g6_concat: input {k} is not on device")
    first = tensors[0]
    rank = len(first.shape)
    if rank not in (3, 4):
        raise ValueError(f"g6_concat: rank must be 3 or 4, got {rank}")
    for k, t in enumerate(tensors):
        if len(t.shape) != rank:
            raise ValueError(f"g6_concat: input {k} has rank {len(t.shape)}, input 0 has rank {rank}")
        if t.dtype != first.dtype:
            raise ValueError(f"g6_concat: input {k} dtype {t.dtype} differs from {first.dtype}")
        if t.layout != first.layout:
            raise ValueError(f"g6_concat: input {k} layout {t.layout} differs from {first.layout}")
        if t.memory_config().is_sharded():
            raise ValueError(f"g6_concat: input {k} is sharded; only interleaved inputs are accepted")
        if t.device() != first.device():
            raise ValueError(f"g6_concat: input {k} is on another device")
    if not isinstance(dim, int) or isinstance(dim, bool) or not -rank <= dim < rank:
        raise ValueError(f"g6_concat: dim {dim!r} outside [{-rank}, {rank - 1}]")
    d = _canonical_dim(dim, rank)
    for k, t in enumerate(tensors):
        for i in range(rank):
            if i != d and int(t.shape[i]) != int(first.shape[i]):
                raise ValueError(
                    f"g6_concat: input {k} shape {list(t.shape)} differs from {list(first.shape)} at dim {i}"
                )
        if any(int(v) == 0 for v in t.shape):
            raise ValueError(f"g6_concat: input {k} has a zero-sized dim {list(t.shape)}")
    if memory_config is not None and memory_config.is_sharded():
        raise ValueError("g6_concat: memory_config must be interleaved")


def _scenario_of_call(tensors, dim, memory_config):
    out_l1 = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
    return {
        "shapes": [list(t.shape) for t in tensors],
        "dim": dim,
        "dtype": tensors[0].dtype,
        "layout": tensors[0].layout,
        "input_buffers": [_buffer_name(t.memory_config().buffer_type) for t in tensors],
        "memory_config": "l1" if out_l1 else None,
    }


def validate(tensors, dim=0, *, memory_config=None):
    _check_structure(tensors, dim, memory_config)
    scenario = _scenario_of_call(list(tensors), dim, memory_config)
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
# Plan report and public entry point
# ---------------------------------------------------------------------------


def _output_memory_config(memory_config):
    return ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config


def select_plan(tensors, dim=0, *, memory_config=None) -> dict:
    """The plan a call with these arguments runs (BB-08): regime, cores, block, depth, roles.

    Depends only on shapes, dtype, layout, placements and grid, never on free L1.
    """
    tensors = list(tensors)
    d = _canonical_dim(dim, len(tensors[0].shape))
    built = built_program(tensors, d, _output_memory_config(memory_config))
    plan = dict(built["plan"])
    plan["roles"] = dict(plan["roles"])
    return plan


def g6_concat(tensors, dim=0, *, memory_config=None) -> ttnn.Tensor:
    validate(tensors, dim, memory_config=memory_config)
    tensors = list(tensors)
    d = _canonical_dim(dim, len(tensors[0].shape))
    out_memory_config = _output_memory_config(memory_config)
    built = built_program(tensors, d, out_memory_config)

    first = tensors[0]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(built["out_shape"]),
        first.dtype,
        first.layout,
        first.device(),
        out_memory_config,
    )
    program = create_program_descriptor(built, tensors, output)
    return ttnn.generic_op([*tensors, output], program)
