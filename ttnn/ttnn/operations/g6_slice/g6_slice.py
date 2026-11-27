# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: x[b_0:e_0:s_0, ..., b_{r-1}:e_{r-1}:s_{r-1}] as one data-movement program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
The taggers read one scenario dict (the form eval/golden_tests/g6_slice/feature_spec.py
uses); validate() builds that dict from the call.

Design: op_design.md. The output pages are split over the grid in contiguous ranges; each
core moves its range in blocks of `block_pages` pages through one CB (reader on NoC0, writer
on NoC1). No compute kernel: every byte of out is a byte of x moved by the NoC.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_slice_program_descriptor import create_program_descriptor

TILE = 32
RANK = 4  # every kernel sees the op as rank 4 (left-padded with size-1 / begin-0 dims)

# --- Blocking knobs (read at call time so a test can override them) -----------------------
# Overlap lamp of op_design.md, measured on BH (110 cores, p1 = case 703): 32-page blocks start the
# writer sooner and run 1.5 % faster than the L1-filling 329 (2534 vs 2573 us); 16..32 is the
# optimum on cases 795 and 1112 too. The L1 budget still clamps it; None -> the L1 bound alone.
BLOCK_PAGES_CAP = 32
CB_DEPTH = 2  # blocks resident in cb_slice_pages: fetch of k+1 overlaps store of k
MAX_CORES = None  # None -> the full compute-with-storage grid
L1_BUDGET_BYTES = 1347776  # BB-05: what the model leaves to the op on every core

CB_ALLOC_ALIGN_BYTES = 64  # per-CB allocation alignment charged against the budget


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def _normalize(size, begin, end, step):
    """PyTorch slice normalization of one dim -> (begin, extent)."""
    if begin < 0:
        begin += size
    if end < 0:
        end += size
    begin = min(max(begin, 0), size)
    end = min(max(end, begin), size)
    return begin, (end - begin + step - 1) // step


def _normalized_begins(scenario):
    if scenario["begins_source"] == "device_tensor":
        return [0] * len(scenario["shape"])
    return [_normalize(n, b, b, 1)[0] for n, b in zip(scenario["shape"], scenario["starts"])]


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


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _memory(tensor):
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


def _structural_checks(input_tensor, starts, ends, steps, slice_dim, num_devices):
    """ValueError for malformed calls; returns (scenario, begins, out_shape)."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_slice: input_tensor must be on device")
    shape = tuple(int(d) for d in input_tensor.shape)
    rank = len(shape)
    if rank < 2 or rank > 4:
        raise ValueError(f"g6_slice: rank must be 2 to 4, got {rank}")

    starts_is_tensor = isinstance(starts, ttnn.Tensor)
    ends_is_tensor = isinstance(ends, ttnn.Tensor)
    if starts_is_tensor != ends_is_tensor:
        raise ValueError("g6_slice: starts and ends must both be lists or both be tensors")

    scenario = {
        "id": "call",
        "model": "call",
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory": _memory(input_tensor),
        "shard": None,
        "slice_dim": slice_dim,
        "num_devices": num_devices,
        "begin_index": None,
    }

    if starts_is_tensor:
        if slice_dim is None or num_devices is None:
            raise ValueError("g6_slice: tensor begins need both slice_dim and num_devices")
        if steps is not None:
            raise ValueError("g6_slice: steps are not accepted with tensor begins")
        if not -rank <= slice_dim <= rank - 1:
            raise ValueError(f"g6_slice: slice_dim {slice_dim} outside [{-rank}, {rank - 1}]")
        sd = slice_dim + rank if slice_dim < 0 else slice_dim
        _check_begins_tensor("starts", starts, rank)
        _check_begins_tensor("ends", ends, rank)
        if num_devices < 1 or shape[sd] % num_devices:
            raise ValueError(f"g6_slice: num_devices {num_devices} does not divide N_{sd} = {shape[sd]}")
        out_shape = list(shape)
        out_shape[sd] = shape[sd] // num_devices
        if any(m == 0 for m in out_shape):
            raise ValueError(f"g6_slice: empty selection, output shape {tuple(out_shape)}")
        scenario.update(
            slice_dim=sd,
            begins_source="device_tensor",
            starts=None,
            ends=None,
            steps=None,
            out_shape=tuple(out_shape),
        )
        return scenario, [0] * rank, out_shape

    if slice_dim is not None or num_devices is not None:
        raise ValueError("g6_slice: slice_dim / num_devices are only accepted with tensor begins")
    starts = [int(v) for v in starts]
    ends = [int(v) for v in ends]
    steps_ = [1] * rank if steps is None else [int(v) for v in steps]
    if len(starts) != rank or len(ends) != rank or len(steps_) != rank:
        raise ValueError(
            f"g6_slice: starts/ends/steps lengths {len(starts)}/{len(ends)}/{len(steps_)} must equal rank {rank}"
        )
    if any(s < 1 for s in steps_):
        raise ValueError(f"g6_slice: every step must be >= 1, got {steps_}")
    begins, out_shape = [], []
    for n, b, e, s in zip(shape, starts, ends, steps_):
        begin, extent = _normalize(n, b, e, s)
        begins.append(begin)
        out_shape.append(extent)
    if any(m == 0 for m in out_shape):
        raise ValueError(f"g6_slice: empty selection, output shape {tuple(out_shape)}")
    scenario.update(
        begins_source="host_list",
        starts=tuple(starts),
        ends=tuple(ends),
        steps=None if steps is None else tuple(steps_),
        out_shape=tuple(out_shape),
    )
    return scenario, begins, out_shape


def validate(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """Structural ValueErrors first, then the SUPPORTED / EXCLUSIONS gate.

    Returns (begins, out_shape) of the normalized selection.
    """
    scenario, begins, out_shape = _structural_checks(input_tensor, starts, ends, steps, slice_dim, num_devices)

    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), dict(axes))

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_slice: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_slice: unsupported combination (refinement candidate): {exc}")

    return begins, out_shape


# ---------------------------------------------------------------------------
# Planning (host): regime, work split, block size
# ---------------------------------------------------------------------------


def _pad4(values, fill):
    return [fill] * (RANK - len(values)) + list(values)


def _select_regime(b4, m4):
    if b4[3] % TILE != 0:
        return "col_realign"
    if b4[2] % TILE == 0 and m4[2] % TILE == 0:
        return "page_copy"
    return "row_runs"


def _core_grid(device):
    grid = device.compute_with_storage_grid_size()
    max_cores = MAX_CORES
    if max_cores is None or max_cores >= grid.x * grid.y:
        return grid, None
    return grid, ttnn.num_cores_to_corerangeset(max_cores, grid, True)


def _split(device, total_out_pages):
    grid, limited = _core_grid(device)
    if limited is None:
        split = ttnn.split_work_to_cores(grid, total_out_pages, row_wise=True)
    else:
        split = ttnn.split_work_to_cores(limited, total_out_pages, row_wise=True)
    num_cores, all_cores, group_1, group_2, pages_1, pages_2 = split
    assignment = []  # (core, core_start_page, core_pages)
    start = 0
    for group, per_core in ((group_1, pages_1), (group_2, pages_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == total_out_pages, (start, total_out_pages)
    return grid, num_cores, all_cores, assignment, pages_1


def _plan(input_tensor, begins, out_shape):
    device = input_tensor.device()
    n4 = _pad4([int(d) for d in input_tensor.shape], 1)
    b4 = _pad4(begins, 0)
    m4 = _pad4(out_shape, 1)

    page_bytes = input_tensor.buffer_page_size()
    if page_bytes != input_tensor.buffer_aligned_page_size():
        raise ValueError("g6_slice: tile pages must be DRAM-aligned")

    regime = _select_regime(b4, m4)
    tensor_out_ht_tiles = math.ceil(m4[2] / TILE)
    tensor_out_wt_tiles = math.ceil(m4[3] / TILE)
    total_out_pages = m4[0] * m4[1] * tensor_out_ht_tiles * tensor_out_wt_tiles

    grid, num_cores, all_cores, assignment, core_pages_max = _split(device, total_out_pages)

    num_cbs = 2 if regime == "col_realign" else 1
    right_col_slots = 1 if regime == "col_realign" else 0
    out_l1_bytes_per_core = 0  # out is DRAM interleaved
    l1_budget = min(L1_BUDGET_BYTES, ttnn.device.get_max_worker_l1_unreserved_size())
    fixed_bytes = out_l1_bytes_per_core + CB_ALLOC_ALIGN_BYTES * num_cbs
    block_pages_cap = (l1_budget - fixed_bytes) // ((CB_DEPTH + right_col_slots) * page_bytes)
    if BLOCK_PAGES_CAP is not None:
        block_pages_cap = min(block_pages_cap, BLOCK_PAGES_CAP)
    if block_pages_cap < 1:
        raise ValueError(f"g6_slice: one page per CB slot exceeds the {l1_budget} B L1 budget (BB-05)")
    block_pages = min(core_pages_max, block_pages_cap)
    num_blocks_max = math.ceil(core_pages_max / block_pages)
    # A core with a single block never has a second block in flight: a deeper ring would never be live.
    cb_depth = min(CB_DEPTH, num_blocks_max)
    l1_bytes_per_core = (cb_depth + right_col_slots) * block_pages * page_bytes + out_l1_bytes_per_core

    return {
        "regime": regime,
        "num_cores": num_cores,
        "block_pages": block_pages,
        "block_pages_cap": block_pages_cap,
        "cb_depth": cb_depth,
        "num_blocks_max": num_blocks_max,
        "l1_bytes_per_core": l1_bytes_per_core,
        "core_roles": {(core.x, core.y): "reader+writer" for core, _, _ in assignment},
        "n4": n4,
        "b4": b4,
        "m4": m4,
        "page_bytes": page_bytes,
        "tensor_in_ht_tiles": math.ceil(n4[2] / TILE),
        "tensor_in_wt_tiles": math.ceil(n4[3] / TILE),
        "tensor_out_ht_tiles": tensor_out_ht_tiles,
        "tensor_out_wt_tiles": tensor_out_wt_tiles,
        "rows_last_tile_row": m4[2] - TILE * (tensor_out_ht_tiles - 1),
        "cols_last_tile_col": m4[3] - TILE * (tensor_out_wt_tiles - 1),
        "first_input_tile_row": b4[2] // TILE,
        "row_offset_in_tile": b4[2] % TILE,
        "first_input_tile_col": b4[3] // TILE,
        "col_offset_in_tile": b4[3] % TILE,
        "total_out_pages": total_out_pages,
        "all_cores": all_cores,
        "assignment": assignment,
        "grid": (grid.x, grid.y),
    }


_REPORT_KEYS = ("regime", "num_cores", "block_pages", "cb_depth", "num_blocks_max", "l1_bytes_per_core", "core_roles")


def describe_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """The plan g6_slice would run for this call (BB-08), without dispatching."""
    begins, out_shape = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)
    plan = _plan(input_tensor, begins, out_shape)
    return {key: plan[key] for key in _REPORT_KEYS}


# ---------------------------------------------------------------------------
# Plan cache (BB-07): a hit patches only the buffer addresses
# ---------------------------------------------------------------------------

_PLAN_CACHE = {}


def _cache_key(input_tensor, begins, out_shape, slice_dim, num_devices):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(int(d) for d in input_tensor.shape),
        tuple(int(d) for d in input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        tuple(_pad4(begins, 0)),
        tuple(_pad4(out_shape, 1)),
        device.id(),
        (grid.x, grid.y),
        BLOCK_PAGES_CAP,
        CB_DEPTH,
        MAX_CORES,
        L1_BUDGET_BYTES,
        slice_dim,
        num_devices,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_slice(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """Return x[starts:ends:steps] (PyTorch semantics) in x's dtype, layout and placement."""
    begins, out_shape = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, begins, out_shape, slice_dim, num_devices)
    cached = _PLAN_CACHE.get(key)
    if cached is None:
        plan = _plan(input_tensor, begins, out_shape)
        descriptor, patch_address = create_program_descriptor(input_tensor, output_tensor, plan)
        _PLAN_CACHE[key] = (descriptor, patch_address)
    else:
        descriptor, patch_address = cached
        patch_address(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
