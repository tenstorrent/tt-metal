# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: copy the elements of x selected by per-dim begin/end/step into a new tensor.

    out[i_0..i_{r-1}] = x[b_0 + s_0*i_0, ..., b_{r-1} + s_{r-1}*i_{r-1}]

One device program (reader -> cb_slice_pages -> writer, no compute kernel) moves only the bytes that hold
selected elements. The op is planned in rank 4 (left-padded with size-1 dims) over output tile pages,
split into contiguous page ranges across the full grid; see op_design.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_slice_program_descriptor import create_program_descriptor, rebind_addresses

# ---------------------------------------------------------------------------
# Block knobs. Read at call time, so a test can override them.
# ---------------------------------------------------------------------------

# Pages per reader/writer handshake. None: the coarsest block the L1 budget allows.
BLOCK_PAGES_CAP = None
# cb_slice_pages slots: the reader fetches block k+1 while the writer stores block k.
CB_DEPTH = 2
# Cores the output-page ranges spread over. None: the full compute grid.
MAX_CORES = None
# L1 per core the model leaves to the op (BB-05).
L1_BUDGET_BYTES = 1347776

TILE = 32
CB_ALLOCATION_ALIGNMENT = 64

# ---------------------------------------------------------------------------
# Scenario dict (the form INPUT_TAGGERS read; same keys as feature_spec.py)
# ---------------------------------------------------------------------------


def _normalize_dim(size, start, end):
    """PyTorch begin/end normalization: negatives count from the end, both clamped into [0, size]."""
    begin = start + size if start < 0 else start
    begin = min(max(begin, 0), size)
    stop = end + size if end < 0 else end
    stop = min(max(stop, begin), size)
    return begin, stop


def _normalized_begins(scenario):
    if scenario["begins_source"] == "device_tensor":
        return (0,) * len(scenario["shape"])
    return tuple(_normalize_dim(n, s, e)[0] for n, s, e in zip(scenario["shape"], scenario["starts"], scenario["ends"]))


def _memory_tag(tensor):
    mc = tensor.memory_config()
    if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if mc.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return "l1_width_sharded"
    return str(mc.memory_layout)


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
    begins = _normalized_begins(scenario)
    out_shape = scenario["out_shape"]
    if begins[-1] % TILE or out_shape[-1] % TILE:
        return "w_non_aligned"
    if begins[-2] % TILE or out_shape[-2] % TILE:
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

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_begins_tensor(name, tensor, rank):
    if tensor.dtype != ttnn.uint32 or tuple(int(d) for d in tensor.shape) != (rank,):
        raise ValueError(
            f"g6_slice: {name} tensor must be uint32 of shape ({rank},), got {tensor.dtype} {tuple(tensor.shape)}"
        )


def _int_list(name, values, rank):
    try:
        values = [int(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"g6_slice: {name} must be a list of ints or a ttnn.Tensor") from exc
    if len(values) != rank:
        raise ValueError(f"g6_slice: {name} has length {len(values)}, input rank is {rank}")
    return values


def _build_scenario(input_tensor, starts, ends, steps, slice_dim, num_devices):
    """Structural checks (ValueError) and the scenario dict of the call."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_slice: input_tensor must be on device")
    shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(shape)
    if rank not in (2, 3, 4):
        raise ValueError(f"g6_slice: input rank must be 2..4, got {rank}")

    scenario = {
        "id": "call",
        "model": "call",
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory": _memory_tag(input_tensor),
        "shard": None,
        "slice_dim": slice_dim,
        "num_devices": num_devices,
        "begin_index": None,
    }

    starts_is_tensor = isinstance(starts, ttnn.Tensor)
    if starts_is_tensor != isinstance(ends, ttnn.Tensor):
        raise ValueError("g6_slice: starts and ends must both be lists or both be tensors")

    if starts_is_tensor:
        if steps is not None:
            raise ValueError("g6_slice: steps cannot be given with tensor starts/ends")
        if slice_dim is None or num_devices is None:
            raise ValueError("g6_slice: tensor starts/ends need both slice_dim and num_devices")
        if not -rank <= slice_dim < rank:
            raise ValueError(f"g6_slice: slice_dim {slice_dim} outside [{-rank}, {rank - 1}]")
        dim = slice_dim + rank if slice_dim < 0 else slice_dim
        _check_begins_tensor("starts", starts, rank)
        _check_begins_tensor("ends", ends, rank)
        if num_devices < 1 or shape[dim] % num_devices:
            raise ValueError(f"g6_slice: num_devices {num_devices} does not divide dim {dim} of size {shape[dim]}")
        out_shape = list(shape)
        out_shape[dim] = shape[dim] // num_devices
        scenario.update(begins_source="device_tensor", starts=None, ends=None, steps=None, slice_dim=dim)
    else:
        if slice_dim is not None or num_devices is not None:
            raise ValueError("g6_slice: slice_dim/num_devices are only valid with tensor starts/ends")
        starts_list = _int_list("starts", starts, rank)
        ends_list = _int_list("ends", ends, rank)
        steps_list = [1] * rank if steps is None else _int_list("steps", steps, rank)
        if any(s < 1 for s in steps_list):
            raise ValueError(f"g6_slice: steps must be >= 1, got {steps_list}")
        out_shape = []
        for size, start, end, step in zip(shape, starts_list, ends_list, steps_list):
            begin, stop = _normalize_dim(size, start, end)
            out_shape.append(-(-(stop - begin) // step))
        scenario.update(
            begins_source="host_list",
            starts=tuple(starts_list),
            ends=tuple(ends_list),
            steps=None if steps is None else tuple(steps_list),
        )

    if any(m == 0 for m in out_shape):
        raise ValueError(f"g6_slice: empty selection, output shape {tuple(out_shape)}")
    scenario["out_shape"] = tuple(out_shape)
    return scenario


def validate(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """Structural checks raise ValueError; support refusals raise UnsupportedAxisValue / ExcludedCell.

    Returns the scenario dict of the call.
    """
    scenario = _build_scenario(input_tensor, starts, ends, steps, slice_dim, num_devices)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_slice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_slice: unsupported combination (refinement candidate): {exc}")
    return scenario


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    regime: str
    row_runs: bool
    w_shift: bool
    N4: tuple
    M4: tuple
    b4: tuple
    tensor_in_ht_tiles: int
    tensor_in_wt_tiles: int
    tensor_out_ht_tiles: int
    tensor_out_wt_tiles: int
    rows_last_tile_row: int
    cols_last_tile_col: int
    page_bytes: int
    elem_bytes: int
    total_out_pages: int
    num_cores: int
    all_cores: object
    # (core, core_start_page, core_pages, core_start_coord (o0, o1, th, tw))
    assignment: tuple
    core_pages_max: int
    block_pages_cap: int
    block_pages: int
    cb_depth: int
    num_blocks_max: int
    l1_bytes_per_core: int


def _rank4(scenario):
    shape = scenario["shape"]
    pad = 4 - len(shape)
    begins = _normalized_begins(scenario)
    return (1,) * pad + shape, (0,) * pad + begins, (1,) * pad + scenario["out_shape"]


def _select_regime(b4, M4):
    row_offset = b4[2] % TILE
    col_offset = b4[3] % TILE
    if col_offset != 0:
        return "col_realign"
    if row_offset == 0 and M4[2] % TILE == 0:
        return "page_copy"
    return "row_runs"


def _out_l1_bytes_per_core(input_tensor):
    # Phase 0 out lives in DRAM; an L1 out (refinement) takes its per-core share out of the budget here.
    if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM:
        return 0
    raise NotImplementedError("g6_slice: L1 placement of out is not planned yet")


def _work_grid(device):
    grid = device.compute_with_storage_grid_size()
    if MAX_CORES is None:
        return grid
    num = min(MAX_CORES, grid.x * grid.y)
    return ttnn.num_cores_to_corerangeset(num, grid, True)


def _plan(input_tensor, scenario):
    N4, b4, M4 = _rank4(scenario)
    regime = _select_regime(b4, M4)
    w_shift = regime == "col_realign"
    row_runs = regime != "page_copy"

    in_ht, in_wt = math.ceil(N4[2] / TILE), math.ceil(N4[3] / TILE)
    out_ht, out_wt = math.ceil(M4[2] / TILE), math.ceil(M4[3] / TILE)
    total_out_pages = M4[0] * M4[1] * out_ht * out_wt
    page_bytes = input_tensor.buffer_page_size()

    num_cores, all_cores, group_1, group_2, pages_1, pages_2 = ttnn.split_work_to_cores(
        _work_grid(input_tensor.device()), total_out_pages, True
    )
    assignment = []
    start = 0
    for group, pages in ((group_1, pages_1), (group_2, pages_2)):
        if pages == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            rest, tw = divmod(start, out_wt)
            rest, th = divmod(rest, out_ht)
            o0, o1 = divmod(rest, M4[1])
            assignment.append((core, start, pages, (o0, o1, th, tw)))
            start += pages
    assert start == total_out_pages
    core_pages_max = pages_1

    num_cbs = 2 if w_shift else 1
    slots_per_page = CB_DEPTH + (1 if w_shift else 0)
    out_l1 = _out_l1_bytes_per_core(input_tensor)
    l1_budget = min(L1_BUDGET_BYTES, ttnn.device.get_max_worker_l1_unreserved_size())
    fixed_bytes = out_l1 + CB_ALLOCATION_ALIGNMENT * num_cbs
    l1_cap = (l1_budget - fixed_bytes) // (slots_per_page * page_bytes)
    block_pages_cap = l1_cap if BLOCK_PAGES_CAP is None else min(BLOCK_PAGES_CAP, l1_cap)
    if block_pages_cap < 1:
        raise RuntimeError(f"g6_slice: one {page_bytes}-B page per slot exceeds the {l1_budget}-B L1 budget")
    block_pages = min(core_pages_max, block_pages_cap)

    return Plan(
        regime=regime,
        row_runs=row_runs,
        w_shift=w_shift,
        N4=N4,
        M4=M4,
        b4=b4,
        tensor_in_ht_tiles=in_ht,
        tensor_in_wt_tiles=in_wt,
        tensor_out_ht_tiles=out_ht,
        tensor_out_wt_tiles=out_wt,
        rows_last_tile_row=M4[2] - TILE * (out_ht - 1),
        cols_last_tile_col=M4[3] - TILE * (out_wt - 1),
        page_bytes=page_bytes,
        elem_bytes=input_tensor.element_size(),
        total_out_pages=total_out_pages,
        num_cores=num_cores,
        all_cores=all_cores,
        assignment=tuple(assignment),
        core_pages_max=core_pages_max,
        block_pages_cap=block_pages_cap,
        block_pages=block_pages,
        cb_depth=CB_DEPTH,
        num_blocks_max=math.ceil(core_pages_max / block_pages),
        l1_bytes_per_core=slots_per_page * block_pages * page_bytes + out_l1,
    )


# Built ProgramDescriptors, keyed by everything that shapes them (BB-07). A hit rewrites only the buffer
# addresses.
_PROGRAM_CACHE = {}


def _cache_key(input_tensor, scenario):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    N4, b4, M4 = _rank4(scenario)
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.memory_config(),
        b4,
        M4,
        device.id(),
        (grid.x, grid.y),
        BLOCK_PAGES_CAP,
        CB_DEPTH,
        MAX_CORES,
        L1_BUDGET_BYTES,
        scenario["slice_dim"],
        scenario["num_devices"],
    )


def describe_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """The plan a call would run, without dispatching it (BB-08, BB-21)."""
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)
    plan = _plan(input_tensor, scenario)
    return {
        "regime": plan.regime,
        "num_cores": plan.num_cores,
        "block_pages": plan.block_pages,
        "block_pages_cap": plan.block_pages_cap,
        "cb_depth": plan.cb_depth,
        "num_blocks_max": plan.num_blocks_max,
        "l1_bytes_per_core": plan.l1_bytes_per_core,
        "total_out_pages": plan.total_out_pages,
        "core_pages_max": plan.core_pages_max,
        "core_roles": {(core.x, core.y): "reader+writer" for core, *_ in plan.assignment},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_slice(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(scenario["out_shape"]),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, scenario)
    program_descriptor = _PROGRAM_CACHE.get(key)
    if program_descriptor is None:
        plan = _plan(input_tensor, scenario)
        program_descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
        _PROGRAM_CACHE[key] = program_descriptor
    else:
        rebind_addresses(program_descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
