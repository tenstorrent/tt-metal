# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat — join K interleaved device tensors along one dim, bit for bit, in ONE device program.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below; the public entry point
calls validate() first. The data path (op_design.md, Blocking Model) is output-stationary over the
linear output-page axis: the reader of each core pulls its contiguous range of output pages from
their source pages (page_relocation) or assembles them from row pieces (row_gather), and the
writer stores them. No compute kernel: values move as bytes only.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_concat_program_descriptor import K_MAX, build_program, select_plan


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is one scenario dict:
#    {"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}
# ---------------------------------------------------------------------------


def _canonical_dim(dim, rank):
    return dim + rank if dim < 0 else dim


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
    bufs = set(inputs["input_buffers"])
    if len(bufs) > 1:
        return "mixed"
    return bufs.pop()


def tag_output_buffer(inputs, axes):
    return "l1" if inputs.get("memory_config") == "l1" else "dram"


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "concat_dim": tag_concat_dim,
    "alignment": tag_alignment,
    "input_buffer": tag_input_buffer,
    "output_buffer": tag_output_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED (Phase 0)
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
# 3. EXCLUSIONS (Phase 0: none)
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


def _structural_check(tensors, dim):
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        raise ValueError("g6_concat: expected a non-empty list of tensors")
    if len(tensors) > K_MAX:
        raise ValueError(f"g6_concat: at most {K_MAX} inputs are supported, got {len(tensors)}")
    shapes = [list(t.shape) for t in tensors]
    rank = len(shapes[0])
    if rank not in (3, 4):
        raise ValueError(f"g6_concat: rank must be 3 or 4, got {rank}")
    for i, (t, s) in enumerate(zip(tensors, shapes)):
        if len(s) != rank:
            raise ValueError(f"g6_concat: input {i} has rank {len(s)}, expected {rank}")
        if t.dtype != tensors[0].dtype:
            raise ValueError(f"g6_concat: input {i} dtype {t.dtype} != {tensors[0].dtype}")
        if t.layout != tensors[0].layout:
            raise ValueError(f"g6_concat: input {i} layout {t.layout} != {tensors[0].layout}")
        if t.storage_type() != ttnn.StorageType.DEVICE:
            raise ValueError(f"g6_concat: input {i} must be a device tensor")
    if not isinstance(dim, int) or not (-rank <= dim <= rank - 1):
        raise ValueError(f"g6_concat: dim {dim} outside [{-rank}, {rank - 1}]")
    d = _canonical_dim(dim, rank)
    for i, s in enumerate(shapes):
        for a in range(rank):
            if a != d and s[a] != shapes[0][a]:
                raise ValueError(f"g6_concat: input {i} shape {s} differs from {shapes[0]} off dim {d}")
        if s[d] < 1:
            raise ValueError(f"g6_concat: input {i} has empty extent along dim {d}")
    for i, t in enumerate(tensors):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError(f"g6_concat: input {i} must be interleaved")
    return shapes, d


def scenario_of(tensors, dim, memory_config):
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
    _structural_check(tensors, dim)
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_concat: memory_config must be interleaved")

    scenario = scenario_of(tensors, dim, memory_config)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(scenario, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_concat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_concat: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Plan reporting (BB-08) and the public entry point
# ---------------------------------------------------------------------------

last_plan = None


def plan_for(tensors, dim=0, memory_config=None):
    """The plan (regime, block_pages, cores, ...) a call with these arguments selects."""
    shapes, d = _structural_check(tensors, dim)
    device = tensors[0].device()
    return select_plan(shapes, d, tensors[0].dtype, tensors[0].layout, device.compute_with_storage_grid_size())


def g6_concat(tensors, dim=0, *, memory_config=None):
    """Concatenate `tensors` along `dim` (torch.cat semantics) in one device program."""
    validate(tensors, dim, memory_config=memory_config)
    global last_plan

    tensors = list(tensors)
    rank = len(tensors[0].shape)
    d = _canonical_dim(dim, rank)
    out_shape = list(tensors[0].shape)
    out_shape[d] = sum(int(t.shape[d]) for t in tensors)
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    device = tensors[0].device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        tensors[0].dtype,
        tensors[0].layout,
        device,
        out_mem,
    )

    program, plan = build_program(tensors, d, output)
    last_plan = plan
    g6_concat.last_plan = plan
    return ttnn.generic_op(tensors + [output], program)
