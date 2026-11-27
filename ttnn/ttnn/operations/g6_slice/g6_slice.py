# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: copy the elements of x selected by per-dim begin/end/step into a new tensor.

One device program (reader on NCRISC/NoC0, writer on BRISC/NoC1, no compute kernel) moves only
the bytes that hold selected elements: whole tile pages when the selection is tile-aligned in
dim r-2 (`page_copy`), whole-width face-row runs when a dim r-2 begin or end lies inside a tile
(`row_runs`), plus an in-L1 column realign when a last-dim begin lies inside a tile
(`col_realign`). Design: op_design.md next to this file.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_slice_program_descriptor import create_program_descriptor, set_buffer_addresses

TILE = 32

# Blocking knobs, read at call time so a test can override them (op_design.md, Blocking Model).
BLOCK_PAGES_CAP = None  # None: the coarsest block the L1 budget allows
CB_DEPTH = 2
MAX_CORES = None  # None: the full compute-with-storage grid
L1_BUDGET_BYTES = 1347776  # per core, left to the op by the model (BB-05)

# Bytes the allocator may add per CB to align its base (DRAM alignment).
CB_ALIGNMENT_SLACK_BYTES = 64


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS -- each reads only the scenario dict (also driven by the golden axes.py)
# ---------------------------------------------------------------------------


def _normalize_begin(size, begin):
    if begin < 0:
        begin += size
    return min(max(begin, 0), size)


def _normalize(size, begin, end, step):
    begin = _normalize_begin(size, begin)
    if end < 0:
        end += size
    end = min(max(end, begin), size)
    return begin, (end - begin + step - 1) // step


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_memory_config(inputs, axes):
    return inputs[0]["memory"]


def tag_alignment(inputs, axes):
    s = inputs[0]
    out_shape = s["out_shape"]
    if s["begins_source"] == "device_tensor":
        begins = (0,) * len(out_shape)
    else:
        begins = tuple(_normalize_begin(n, b) for n, b in zip(s["shape"], s["starts"]))
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


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _memory(tensor):
    mc = tensor.memory_config()
    if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if mc.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return "l1_width_sharded"
    return str(mc.memory_layout)


def _shard(tensor):
    mc = tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED or mc.shard_spec is None:
        return None
    spec = mc.shard_spec
    grid = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    return {"grid": grid, "shape": tuple(spec.shape), "orientation": str(spec.orientation).split(".")[-1]}


def _int_list(name, values, rank):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"g6_slice: {name} must be a list of {rank} ints or a ttnn.Tensor")
    if len(values) != rank:
        raise ValueError(f"g6_slice: {name} has {len(values)} entries, input rank is {rank}")
    return tuple(int(v) for v in values)


def _check_begins_tensor(name, tensor, rank):
    if tensor.dtype != ttnn.uint32 or tuple(tensor.shape) != (rank,):
        raise ValueError(
            f"g6_slice: {name} tensor must be uint32 of shape ({rank},), got {tensor.dtype} {tuple(tensor.shape)}"
        )


def _scenario(input_tensor, starts, ends, steps, slice_dim, num_devices):
    """Structural checks (ValueError), then the scenario dict of feature_spec.py."""
    shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(shape)
    if rank not in (2, 3, 4):
        raise ValueError(f"g6_slice: input rank must be 2, 3 or 4, got {rank}")

    starts_is_tensor = isinstance(starts, ttnn.Tensor)
    if starts_is_tensor != isinstance(ends, ttnn.Tensor):
        raise ValueError("g6_slice: starts and ends must both be lists or both be tensors")

    scenario = {
        "id": "call",
        "model": "call",
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory": _memory(input_tensor),
        "shard": _shard(input_tensor),
        "begin_index": None,
    }

    if starts_is_tensor:
        if steps is not None:
            raise ValueError("g6_slice: steps are not given with tensor starts/ends")
        if slice_dim is None or num_devices is None:
            raise ValueError("g6_slice: tensor starts/ends need both slice_dim and num_devices")
        slice_dim = int(slice_dim)
        if not -rank <= slice_dim < rank:
            raise ValueError(f"g6_slice: slice_dim {slice_dim} outside [{-rank}, {rank - 1}]")
        slice_dim = slice_dim + rank if slice_dim < 0 else slice_dim
        _check_begins_tensor("starts", starts, rank)
        _check_begins_tensor("ends", ends, rank)
        num_devices = int(num_devices)
        if num_devices < 1 or shape[slice_dim] % num_devices:
            raise ValueError(f"g6_slice: num_devices {num_devices} does not divide N_{slice_dim} = {shape[slice_dim]}")
        out_shape = list(shape)
        out_shape[slice_dim] = shape[slice_dim] // num_devices
        scenario.update(
            begins_source="device_tensor",
            starts=None,
            ends=None,
            steps=None,
            slice_dim=slice_dim,
            num_devices=num_devices,
            out_shape=tuple(out_shape),
        )
        return scenario

    if slice_dim is not None or num_devices is not None:
        raise ValueError("g6_slice: slice_dim and num_devices are only given with tensor starts/ends")
    starts = _int_list("starts", starts, rank)
    ends = _int_list("ends", ends, rank)
    steps_ = (1,) * rank if steps is None else _int_list("steps", steps, rank)
    if any(s < 1 for s in steps_):
        raise ValueError(f"g6_slice: every step must be >= 1, got {steps_}")
    out_shape = tuple(_normalize(n, b, e, s)[1] for n, b, e, s in zip(shape, starts, ends, steps_))
    if any(m == 0 for m in out_shape):
        raise ValueError(f"g6_slice: the selection is empty, output shape {out_shape}")
    scenario.update(
        begins_source="host_list",
        starts=starts,
        ends=ends,
        steps=None if steps is None else steps_,
        slice_dim=None,
        num_devices=None,
        out_shape=out_shape,
    )
    return scenario


def validate(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """Structural errors raise ValueError; cells outside SUPPORTED / in EXCLUSIONS raise a
    SupportRefusal. Returns the scenario dict of the call."""
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

    return scenario


# ---------------------------------------------------------------------------
# Planning: regime, work split, block extent
# ---------------------------------------------------------------------------


def _div_up(a, b):
    return -(-a // b)


def _pad4(values, fill):
    return (fill,) * (4 - len(values)) + tuple(values)


def select_regime(row_offset_in_tile, col_offset_in_tile, out_rows):
    if col_offset_in_tile != 0:
        return "col_realign"
    if row_offset_in_tile == 0 and out_rows % TILE == 0:
        return "page_copy"
    return "row_runs"


def _core_set(grid):
    if MAX_CORES is None:
        return grid
    return ttnn.num_cores_to_corerangeset(min(int(MAX_CORES), grid.x * grid.y), grid, True)


def _plan(input_tensor, scenario):
    """Host-list TILE plan: everything the descriptor needs, derived from the knobs once."""
    device = input_tensor.device()
    shape4 = _pad4(scenario["shape"], 1)
    out4 = _pad4(scenario["out_shape"], 1)
    begins4 = _pad4(tuple(_normalize_begin(n, b) for n, b in zip(scenario["shape"], scenario["starts"])), 0)

    tensor_in_ht_tiles = _div_up(shape4[2], TILE)
    tensor_in_wt_tiles = _div_up(shape4[3], TILE)
    tensor_out_ht_tiles = _div_up(out4[2], TILE)
    tensor_out_wt_tiles = _div_up(out4[3], TILE)
    total_out_pages = out4[0] * out4[1] * tensor_out_ht_tiles * tensor_out_wt_tiles

    row_offset_in_tile = begins4[2] % TILE
    col_offset_in_tile = begins4[3] % TILE
    regime = select_regime(row_offset_in_tile, col_offset_in_tile, out4[2])

    grid = device.compute_with_storage_grid_size()
    num_cores, all_cores, core_group_1, core_group_2, pages_g1, pages_g2 = ttnn.split_work_to_cores(
        _core_set(grid), total_out_pages, row_wise=True
    )
    core_pages_max = pages_g1

    page_bytes = input_tensor.buffer_page_size()
    cb_depth = int(CB_DEPTH)
    w_shift = regime == "col_realign"
    num_cbs = 2 if w_shift else 1
    slots_per_page = cb_depth + (1 if w_shift else 0)
    l1_budget = min(int(L1_BUDGET_BYTES), ttnn.device.get_max_worker_l1_unreserved_size())
    out_l1_bytes_per_core = 0  # out is DRAM interleaved in every SUPPORTED cell
    fixed_bytes = out_l1_bytes_per_core + CB_ALIGNMENT_SLACK_BYTES * num_cbs
    block_pages_cap = (l1_budget - fixed_bytes) // (slots_per_page * page_bytes)
    if BLOCK_PAGES_CAP is not None:
        block_pages_cap = min(block_pages_cap, int(BLOCK_PAGES_CAP))
    if block_pages_cap < 1:
        raise RuntimeError(
            f"g6_slice: one page per CB slot ({slots_per_page} x {page_bytes} B) does not fit the "
            f"{l1_budget - fixed_bytes} B L1 budget"
        )
    block_pages = min(core_pages_max, block_pages_cap)

    # Per-core contiguous output-page ranges, start coordinate decomposed on the host.
    assignment = []
    start = 0
    for group, pages in ((core_group_1, pages_g1), (core_group_2, pages_g2)):
        if pages == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            tw = start % tensor_out_wt_tiles
            rest = start // tensor_out_wt_tiles
            th = rest % tensor_out_ht_tiles
            rest //= tensor_out_ht_tiles
            assignment.append(
                {"core": core, "start_page": start, "pages": pages, "coord": (rest // out4[1], rest % out4[1], th, tw)}
            )
            start += pages
    assert start == total_out_pages

    return {
        "regime": regime,
        "shape4": shape4,
        "out4": out4,
        "begins4": begins4,
        "tensor_in_ht_tiles": tensor_in_ht_tiles,
        "tensor_in_wt_tiles": tensor_in_wt_tiles,
        "tensor_out_ht_tiles": tensor_out_ht_tiles,
        "tensor_out_wt_tiles": tensor_out_wt_tiles,
        "rows_last_tile_row": out4[2] - TILE * (tensor_out_ht_tiles - 1),
        "cols_last_tile_col": out4[3] - TILE * (tensor_out_wt_tiles - 1),
        "total_out_pages": total_out_pages,
        "row_offset_in_tile": row_offset_in_tile,
        "col_offset_in_tile": col_offset_in_tile,
        "first_input_tile_row": begins4[2] // TILE,
        "first_input_tile_col": begins4[3] // TILE,
        "num_cores": num_cores,
        "all_cores": all_cores,
        "assignment": assignment,
        "core_pages_max": core_pages_max,
        "page_bytes": page_bytes,
        "cb_depth": cb_depth,
        "block_pages": block_pages,
        "block_pages_cap": block_pages_cap,
        "num_blocks_max": _div_up(core_pages_max, block_pages),
        "l1_bytes_per_core": slots_per_page * block_pages * page_bytes + out_l1_bytes_per_core,
    }


# Built descriptors, keyed on everything that shapes the program or its per-core arguments
# (BB-07). A hit only rewrites the buffer-address slots.
_PROGRAM_CACHE = {}


def _cache_key(input_tensor, scenario):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    begins = tuple(_normalize_begin(n, b) for n, b in zip(scenario["shape"], scenario["starts"]))
    return (
        scenario["shape"],
        tuple(int(d) for d in input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        begins,
        scenario["out_shape"],
        device.id(),
        (grid.x, grid.y),
        BLOCK_PAGES_CAP,
        CB_DEPTH,
        MAX_CORES,
        L1_BUDGET_BYTES,
    )


def describe_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """The plan a call would run (BB-08, BB-21), without dispatching."""
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)
    plan = _plan(input_tensor, scenario)
    return {
        "regime": plan["regime"],
        "num_cores": plan["num_cores"],
        "block_pages": plan["block_pages"],
        "cb_depth": plan["cb_depth"],
        "num_blocks_max": plan["num_blocks_max"],
        "l1_bytes_per_core": plan["l1_bytes_per_core"],
        "core_roles": {(a["core"].x, a["core"].y): "reader+writer" for a in plan["assignment"]},
        "total_out_pages": plan["total_out_pages"],
        "core_pages_max": plan["core_pages_max"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_slice(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(scenario["out_shape"])),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, scenario)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        cached = create_program_descriptor(input_tensor, output_tensor, _plan(input_tensor, scenario))
        _PROGRAM_CACHE[key] = cached
    else:
        set_buffer_addresses(cached, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], cached.descriptor)
