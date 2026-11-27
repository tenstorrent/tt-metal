# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: copy x[b_0 : e_0 : s_0, ..., b_{r-1} : e_{r-1} : s_{r-1}] into a new tensor in one program.

The output takes x's dtype, layout and memory configuration; every element is a bit copy of an
element of x. Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_slice_program_descriptor import create_program_descriptor, make_plan, patch_buffer_addresses

# Blocking knobs, read at call time so a test can override them.
BLOCK_PAGES_CAP = None  # None: the coarsest block the L1 budget allows
CB_DEPTH = 2
MAX_CORES = None  # None: the full compute_with_storage_grid_size()
L1_BUDGET_BYTES = 1347776  # what the model leaves free per core (BB-05)

TILE = 32


# ---------------------------------------------------------------------------
# Scenario helpers (the dict feature_spec.py and axes.classify_call build)
# ---------------------------------------------------------------------------


def _normalize_dim(size, begin, end, step):
    """PyTorch slice semantics: (begin, length)."""
    if begin < 0:
        begin += size
    if end < 0:
        end += size
    begin = min(max(begin, 0), size)
    end = min(max(end, begin), size)
    return begin, (end - begin + step - 1) // step


def _begins(scenario):
    """Normalized begins; the device form's begin is a whole tile by definition (b = 0 for tagging)."""
    shape = scenario["shape"]
    if scenario["begins_source"] == "device_tensor":
        return (0,) * len(shape)
    steps = scenario["steps"] or (1,) * len(shape)
    return tuple(
        _normalize_dim(n, b, e, s)[0] for n, b, e, s in zip(shape, scenario["starts"], scenario["ends"], steps)
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_memory_config(inputs, axes):
    return inputs[0]["memory"]


def tag_alignment(inputs, axes):
    scenario = inputs[0]
    b = _begins(scenario)
    m = scenario["out_shape"]
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
    return "unit_step" if steps is None or all(s == 1 for s in steps) else "strided"


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
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "begins_source": ["host_list"],
    "step_mode": ["unit_step"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _memory_label(tensor):
    mc = tensor.memory_config()
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return "l1_width_sharded"
    if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if mc.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    return str(mc.memory_layout)


def _check_begins_tensor(name, tensor, rank):
    if tensor.dtype != ttnn.uint32 or tuple(int(d) for d in tensor.shape) != (rank,):
        raise ValueError(
            f"g6_slice: {name} tensor must be uint32 of shape ({rank},), got {tensor.dtype} {tuple(tensor.shape)}"
        )


def _scenario(input_tensor, starts, ends, steps, slice_dim, num_devices):
    """Structural checks (ValueError), then the scenario dict the taggers read."""
    shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(shape)
    if rank < 2 or rank > 4:
        raise ValueError(f"g6_slice: input rank must be 2 to 4, got {rank}")

    starts_is_tensor = isinstance(starts, ttnn.Tensor)
    if starts_is_tensor != isinstance(ends, ttnn.Tensor):
        raise ValueError("g6_slice: starts and ends must both be lists or both be tensors")

    scenario = {
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory": _memory_label(input_tensor),
        "slice_dim": slice_dim,
        "num_devices": num_devices,
        "begin_index": None,
    }

    if starts_is_tensor:
        if steps is not None:
            raise ValueError("g6_slice: steps must be None with tensor starts/ends")
        if slice_dim is None or num_devices is None:
            raise ValueError("g6_slice: tensor starts/ends need both slice_dim and num_devices")
        if not -rank <= slice_dim <= rank - 1:
            raise ValueError(f"g6_slice: slice_dim {slice_dim} outside [{-rank}, {rank - 1}]")
        dim = slice_dim + rank if slice_dim < 0 else slice_dim
        _check_begins_tensor("starts", starts, rank)
        _check_begins_tensor("ends", ends, rank)
        if num_devices < 1 or shape[dim] % num_devices:
            raise ValueError(f"g6_slice: num_devices {num_devices} does not divide dim {dim} of size {shape[dim]}")
        out_shape = list(shape)
        out_shape[dim] = shape[dim] // num_devices
        scenario.update(
            begins_source="device_tensor",
            starts=None,
            ends=None,
            steps=None,
            slice_dim=dim,
            out_shape=tuple(out_shape),
        )
    else:
        if slice_dim is not None or num_devices is not None:
            raise ValueError("g6_slice: slice_dim/num_devices are only valid with tensor starts/ends")
        starts, ends = tuple(int(v) for v in starts), tuple(int(v) for v in ends)
        if len(starts) != rank or len(ends) != rank:
            raise ValueError(f"g6_slice: starts/ends must have length {rank}, got {len(starts)}/{len(ends)}")
        if steps is not None:
            steps = tuple(int(s) for s in steps)
            if len(steps) != rank:
                raise ValueError(f"g6_slice: steps must have length {rank}, got {len(steps)}")
            if any(s < 1 for s in steps):
                raise ValueError(f"g6_slice: every step must be >= 1, got {steps}")
        unit = steps or (1,) * rank
        out_shape = tuple(_normalize_dim(n, b, e, s)[1] for n, b, e, s in zip(shape, starts, ends, unit))
        if any(m == 0 for m in out_shape):
            raise ValueError(f"g6_slice: empty selection, output shape {out_shape}")
        scenario.update(begins_source="host_list", starts=starts, ends=ends, steps=steps, out_shape=out_shape)
    return scenario


def validate(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """Structural errors raise ValueError; cells outside the support contract raise a SupportRefusal.

    Returns the scenario dict (normalized begins under "begins") for the caller's planning."""
    scenario = _scenario(input_tensor, starts, ends, steps, slice_dim, num_devices)

    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_slice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_slice: unsupported combination (refinement candidate): {exc}")

    scenario["begins"] = _begins(scenario)
    return scenario


# ---------------------------------------------------------------------------
# Planning and the plan cache (BB-07)
# ---------------------------------------------------------------------------

# key -> [SlicePlan, ProgramDescriptor or None]
_PLAN_CACHE = {}


def _pad4(values, fill):
    return tuple([fill] * (4 - len(values)) + list(values))


def _plan(input_tensor, scenario):
    begins4 = _pad4(scenario["begins"], 0)
    out_shape4 = _pad4(scenario["out_shape"], 1)
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    key = (
        scenario["shape"],
        tuple(int(d) for d in input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        repr(input_tensor.memory_config()),
        begins4,
        out_shape4,
        device.id(),
        (grid.x, grid.y),
        BLOCK_PAGES_CAP,
        CB_DEPTH,
        MAX_CORES,
        L1_BUDGET_BYTES,
        scenario["slice_dim"],
        scenario["num_devices"],
    )
    entry = _PLAN_CACHE.get(key)
    if entry is None:
        plan = make_plan(
            input_tensor,
            begins4,
            out_shape4,
            block_pages_cap=BLOCK_PAGES_CAP,
            cb_depth=CB_DEPTH,
            max_cores=MAX_CORES,
            l1_budget_bytes=L1_BUDGET_BYTES,
        )
        entry = _PLAN_CACHE[key] = [plan, None]
    return entry


def describe_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """The plan a call selects (BB-08, BB-21), without dispatching."""
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)
    plan, _ = _plan(input_tensor, scenario)
    return plan.report()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_slice(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)

    entry = _plan(input_tensor, scenario)
    plan = entry[0]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(scenario["out_shape"])),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    if entry[1] is None:
        entry[1] = create_program_descriptor(input_tensor, output_tensor, plan)
    else:
        patch_buffer_addresses(entry[1], input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], entry[1])
