# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: PyTorch basic-indexing slice x[b0:e0:s0, ..., b_{r-1}:e_{r-1}:s_{r-1}] in ONE device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().

Phase 0: bfloat16, TILE, DRAM interleaved, host-list begins, unit steps, every alignment
(R1 aligned tile copy, R2 shifted tile gather; op_design.md -> Regimes).

Program caching (BB-07): the ProgramDescriptor is built once per program signature (shapes,
dtype, layout, placement, regime flags, device) and cached here; every value that changes
between calls with the same signature (buffer addresses, begins) lives in COMMON runtime args,
which a cache hit patches in place. Begins are never compile-time values.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_slice_program_descriptor import (
    build_plan,
    create_program_descriptor,
    patch_common_runtime_args,
)


# ---------------------------------------------------------------------------
# helpers: argument normalization (PyTorch semantics)
# ---------------------------------------------------------------------------

TILE = 32


def _normalize_dim(size, begin, end, step):
    if begin < 0:
        begin += size
    if end < 0:
        end += size
    begin = min(max(begin, 0), size)
    end = min(max(end, begin), size)
    return begin, end, (end - begin + step - 1) // step


def _memory_name(tensor):
    mc = tensor.memory_config()
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return "l1_width_sharded"
    if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if mc.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    return str(mc.memory_layout)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs = (scenario_dict,), see feature_spec.py)
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_memory_config(inputs, axes):
    return inputs[0]["memory"]


def tag_alignment(inputs, axes):
    s = inputs[0]
    m = s["out_shape"]
    if s["begins_source"] == "device_tensor" or s.get("starts") is None:
        b = (0,) * len(m)
    else:
        shape = s["shape"]
        steps = s["steps"] or (1,) * len(shape)
        b = tuple(_normalize_dim(shape[d], s["starts"][d], s["ends"][d], steps[d])[0] for d in range(len(shape)))
    if b[-1] % TILE or m[-1] % TILE:
        return "w_non_aligned"
    if b[-2] % TILE or m[-2] % TILE:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0]["shape"])


def tag_begins_source(inputs, axes):
    return inputs[0]["begins_source"]


def tag_step_mode(inputs, axes):
    steps = inputs[0]["steps"]
    return "unit_step" if steps is None or all(int(v) == 1 for v in steps) else "strided"


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "memory_config": tag_memory_config,
    "alignment": tag_alignment,
    "rank": tag_rank,
    "begins_source": tag_begins_source,
    "step_mode": tag_step_mode,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["dram_interleaved"],
    # "h_non_aligned" must never be excluded or split off (spec rule).
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "begins_source": ["host_list"],
    "step_mode": ["unit_step"],
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


def _build_scenario(input_tensor, starts, ends, steps, slice_dim, num_devices):
    """Argument checks (ValueError) + the scenario dict axes.classify_call builds."""
    shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(shape)
    device_form = isinstance(starts, ttnn.Tensor) or isinstance(ends, ttnn.Tensor)
    scenario = {
        "id": "call",
        "model": "call",
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory": _memory_name(input_tensor),
        "shard": None,
        "begins_source": "device_tensor" if device_form else "host_list",
        "slice_dim": slice_dim,
        "num_devices": num_devices,
        "begin_index": None,
    }
    if device_form:
        if not (isinstance(starts, ttnn.Tensor) and isinstance(ends, ttnn.Tensor)):
            raise ValueError("g6_slice: starts and ends must both be tensors or both be lists")
        if slice_dim is None or num_devices is None:
            raise ValueError("g6_slice: tensor begins need both slice_dim and num_devices")
        if steps is not None:
            raise ValueError("g6_slice: steps are not given with tensor begins")
        if not -rank <= slice_dim < rank:
            raise ValueError(f"g6_slice: slice_dim {slice_dim} outside [-{rank}, {rank - 1}]")
        sd = slice_dim % rank
        for name, t in (("starts", starts), ("ends", ends)):
            if t.dtype != ttnn.uint32 or tuple(int(v) for v in t.shape) != (rank,):
                raise ValueError(f"g6_slice: {name} tensor must be uint32 of shape ({rank},)")
        if num_devices < 1 or shape[sd] % num_devices:
            raise ValueError(f"g6_slice: num_devices {num_devices} does not divide N_{sd} = {shape[sd]}")
        out_shape = list(shape)
        out_shape[sd] = shape[sd] // num_devices
        scenario.update(slice_dim=sd, starts=None, ends=None, steps=None, out_shape=tuple(out_shape))
        scenario["_begins"] = None
        return scenario

    if slice_dim is not None or num_devices is not None:
        raise ValueError("g6_slice: slice_dim / num_devices are only given with tensor begins")
    starts = [int(v) for v in starts]
    ends = [int(v) for v in ends]
    if len(starts) != rank:
        raise ValueError(f"g6_slice: len(starts)={len(starts)} != rank {rank}")
    if len(ends) != rank:
        raise ValueError(f"g6_slice: len(ends)={len(ends)} != rank {rank}")
    steps_ = [1] * rank if steps is None else [int(v) for v in steps]
    if len(steps_) != rank:
        raise ValueError(f"g6_slice: len(steps)={len(steps_)} != rank {rank}")
    if any(s < 1 for s in steps_):
        raise ValueError(f"g6_slice: every step must be >= 1, got {steps_}")
    begins, out_shape = [], []
    for d in range(rank):
        b, _, m = _normalize_dim(shape[d], starts[d], ends[d], steps_[d])
        if m == 0:
            raise ValueError(f"g6_slice: empty selection on dim {d}")
        begins.append(b)
        out_shape.append(m)
    scenario.update(
        starts=tuple(starts),
        ends=tuple(ends),
        steps=None if steps is None else tuple(steps_),
        out_shape=tuple(out_shape),
    )
    scenario["_begins"] = tuple(begins)
    return scenario


def validate(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """Argument checks (ValueError), then the SUPPORTED / EXCLUSIONS gate. Returns the scenario."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError("g6_slice: input_tensor must be a ttnn.Tensor")
    rank = len(input_tensor.shape)
    if not 2 <= rank <= 4:
        raise ValueError(f"g6_slice: rank must be 2..4, got {rank}")
    scenario = _build_scenario(input_tensor, starts, ends, steps, slice_dim, num_devices)

    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_slice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_slice: unsupported combination (refinement candidate): {exc}")
    return scenario


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}
_LAST_PLAN = None


def last_plan():
    """The plan (regime, flags, block knobs, core split) the last call selected (BB-08)."""
    return dict(_LAST_PLAN) if _LAST_PLAN is not None else None


def g6_slice(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    global _LAST_PLAN
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)

    device = input_tensor.device()
    out_shape = scenario["out_shape"]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    plan = build_plan(input_tensor, scenario["_begins"], out_shape)
    key = (
        id(device),
        tuple(int(d) for d in input_tensor.padded_shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        plan["signature"],
    )
    cached = _DESCRIPTOR_CACHE.get(key)
    if cached is None:
        cached = create_program_descriptor(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = cached
    descriptor, program_report = cached
    patch_common_runtime_args(descriptor, input_tensor, output_tensor, plan)
    _LAST_PLAN = {**plan["report"], **program_report}
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
