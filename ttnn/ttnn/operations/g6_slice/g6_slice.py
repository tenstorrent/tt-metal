# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice: copy the elements of `x` selected by per-dim begin/end/step into a new tensor.

One device program per call (reader NCRISC/NoC0 + writer BRISC/NoC1, no compute kernel). The
flat output-page index is split into contiguous per-core ranges over the full grid; each core
walks its range in blocks of `block_pages` (see g6_slice_program_descriptor.plan_blocks).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_slice_program_descriptor import (
    REGIME_ELEMENT_SHIFT,
    REGIME_PAGE_COPY,
    REGIME_ROW_SHIFT,
    TILE,
    create_program_descriptor,
    plan_blocks,
)

_OP = "g6_slice"


def _normalize(size, begin, end, step):
    """PyTorch basic-indexing normalization of one dim -> (begin, extent)."""
    if begin < 0:
        begin += size
    if end < 0:
        end += size
    begin = min(max(begin, 0), size)
    end = min(max(end, begin), size)
    return begin, (end - begin + step - 1) // step


def _memory_label(tensor):
    mc = tensor.memory_config()
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return "l1_width_sharded"
    if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return "l1_interleaved" if mc.buffer_type == ttnn.BufferType.L1 else "dram_interleaved"
    return str(mc.memory_layout)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads the scenario dict of feature_spec.py (inputs[0])
# ---------------------------------------------------------------------------


def _scenario_begins(s):
    """Normalized begins and output shape of a scenario (device form: begins 0, see tag_alignment)."""
    shape = s["shape"]
    if s["begins_source"] == "device_tensor":
        return [0] * len(shape), list(s["out_shape"])
    steps = s["steps"] or (1,) * len(shape)
    begins, extents = [], []
    for size, b, e, st in zip(shape, s["starts"], s["ends"], steps):
        nb, m = _normalize(size, b, e, st)
        begins.append(nb)
        extents.append(m)
    return begins, extents


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_memory_config(inputs, axes):
    return inputs[0]["memory"]


def tag_alignment(inputs, axes):
    begins, extents = _scenario_begins(inputs[0])
    if begins[-1] % TILE or extents[-1] % TILE:
        return "w_non_aligned"
    if begins[-2] % TILE or extents[-2] % TILE:
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


def _check_call(input_tensor, starts, ends, steps, slice_dim, num_devices):
    """Malformed calls -> ValueError (not support refusals). Returns the canonical slice_dim."""
    rank = len(input_tensor.shape)
    starts_is_tensor = isinstance(starts, ttnn.Tensor)
    ends_is_tensor = isinstance(ends, ttnn.Tensor)
    if starts_is_tensor != ends_is_tensor:
        raise ValueError(f"{_OP}: starts and ends must both be lists or both be tensors")

    if not starts_is_tensor:
        if slice_dim is not None or num_devices is not None:
            raise ValueError(f"{_OP}: slice_dim and num_devices are only given with tensor starts and ends")
        if len(starts) != rank or len(ends) != rank:
            raise ValueError(f"{_OP}: starts and ends must have length {rank}, got {len(starts)} and {len(ends)}")
        if steps is not None:
            if len(steps) != rank:
                raise ValueError(f"{_OP}: steps must have length {rank}, got {len(steps)}")
            if any(int(s) < 1 for s in steps):
                raise ValueError(f"{_OP}: every step must be at least 1, got {list(steps)}")
        steps_ = [1] * rank if steps is None else [int(s) for s in steps]
        for d in range(rank):
            _, m = _normalize(int(input_tensor.shape[d]), int(starts[d]), int(ends[d]), steps_[d])
            if m == 0:
                raise ValueError(f"{_OP}: the selection of dim {d} is empty")
        return None

    if slice_dim is None or num_devices is None:
        raise ValueError(f"{_OP}: tensor starts and ends need both slice_dim and num_devices")
    if steps is not None:
        raise ValueError(f"{_OP}: steps are not given with tensor starts and ends")
    if not -rank <= slice_dim < rank:
        raise ValueError(f"{_OP}: slice_dim {slice_dim} outside [{-rank}, {rank - 1}]")
    for name, t in (("starts", starts), ("ends", ends)):
        if t.dtype != ttnn.uint32 or tuple(t.shape) != (rank,):
            raise ValueError(
                f"{_OP}: {name} must be a uint32 tensor of shape ({rank},), got {t.dtype} {tuple(t.shape)}"
            )
    sd = slice_dim % rank
    if num_devices < 1 or int(input_tensor.shape[sd]) % num_devices:
        raise ValueError(f"{_OP}: num_devices {num_devices} does not divide N_{sd} = {int(input_tensor.shape[sd])}")
    return sd


def _scenario_from_call(input_tensor, starts, ends, steps, sd, num_devices):
    """The feature_spec scenario dict of a real call (the form INPUT_TAGGERS read)."""
    shape = tuple(int(d) for d in input_tensor.shape)
    scenario = {
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory": _memory_label(input_tensor),
    }
    if sd is not None or isinstance(starts, ttnn.Tensor):
        out_shape = list(shape)
        out_shape[sd] = shape[sd] // num_devices
        scenario.update(begins_source="device_tensor", starts=None, ends=None, steps=None, out_shape=tuple(out_shape))
    else:
        scenario.update(
            begins_source="host_list",
            starts=tuple(int(v) for v in starts),
            ends=tuple(int(v) for v in ends),
            steps=None if steps is None else tuple(int(v) for v in steps),
        )
    return scenario


def validate(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """ValueError for malformed calls, then the SUPPORTED / EXCLUSIONS gate. Returns the scenario."""
    sd = _check_call(input_tensor, starts, ends, steps, slice_dim, num_devices)
    scenario = _scenario_from_call(input_tensor, starts, ends, steps, sd, num_devices)

    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{_OP}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{_OP}: unsupported combination (refinement candidate): {exc}")
    return scenario


# ---------------------------------------------------------------------------
# Geometry + plan
# ---------------------------------------------------------------------------


def _geometry(input_tensor, scenario):
    """Fold a host-list selection to the canonical (D0, D1, H, W) index space of op_design.md."""
    shape = scenario["shape"]
    rank = len(shape)
    steps = scenario["steps"] or (1,) * rank
    begins, extents = [], []
    for size, b, e, st in zip(shape, scenario["starts"], scenario["ends"], steps):
        nb, m = _normalize(size, b, e, st)
        begins.append(nb)
        extents.append(m)

    lead = 4 - rank
    n4 = (1,) * lead + shape
    b4 = (0,) * lead + tuple(begins)
    m4 = (1,) * lead + tuple(extents)
    padded = input_tensor.padded_shape

    hb, row_offset = divmod(b4[2], TILE)
    wb, col_offset = divmod(b4[3], TILE)
    if col_offset:
        regime = REGIME_ELEMENT_SHIFT
    elif row_offset:
        regime = REGIME_ROW_SHIFT
    else:
        regime = REGIME_PAGE_COPY

    out_h_tiles = -(-m4[2] // TILE)
    out_w_tiles = -(-m4[3] // TILE)
    return {
        "out_shape": tuple(extents),
        "regime": regime,
        "N_D1": n4[1],
        "in_h_tiles": int(padded[-2]) // TILE,
        "in_w_tiles": int(padded[-1]) // TILE,
        "b_D0": b4[0],
        "b_D1": b4[1],
        "hb": hb,
        "wb": wb,
        "row_offset": row_offset,
        "col_offset": col_offset,
        "M_D0": m4[0],
        "M_D1": m4[1],
        "M_H": m4[2],
        "M_W": m4[3],
        "out_h_tiles": out_h_tiles,
        "out_w_tiles": out_w_tiles,
        "total_out_pages": m4[0] * m4[1] * out_h_tiles * out_w_tiles,
        "needs_below_any": row_offset > 0 and m4[2] > TILE - row_offset,
        "needs_right_any": col_offset > 0 and m4[3] > TILE - col_offset,
    }


_REGIME_NAMES = {REGIME_PAGE_COPY: "page_copy", REGIME_ROW_SHIFT: "row_shift", REGIME_ELEMENT_SHIFT: "element_shift"}


def _plan(input_tensor, scenario):
    geometry = _geometry(input_tensor, scenario)
    blocks = plan_blocks(input_tensor.device(), geometry, input_tensor.buffer_aligned_page_size())
    return geometry, blocks


def g6_slice_plan(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    """The plan a call selects (BB-08): regime, block knobs and the per-core role dump."""
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)
    geometry, blocks = _plan(input_tensor, scenario)
    return {
        "regime": _REGIME_NAMES[geometry["regime"]],
        "out_shape": geometry["out_shape"],
        "total_out_pages": geometry["total_out_pages"],
        "num_cores": blocks["num_cores"],
        "pages_per_core": blocks["pages_per_core"],
        "block_pages": blocks["block_pages"],
        "buffer_depth": blocks["buffer_depth"],
        "sources_per_output": blocks["sources_per_output"],
        "assembled_per_output": blocks["assembled_per_output"],
        # Every participating core runs both roles: reader (NCRISC, NoC0) and writer (BRISC, NoC1).
        "cores": [
            {"core": (core.x, core.y), "roles": ("reader", "writer"), "start_page": start, "num_pages": n}
            for core, start, n in blocks["assignment"]
        ],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_slice(input_tensor, starts, ends, steps=None, *, slice_dim=None, num_devices=None):
    scenario = validate(input_tensor, starts, ends, steps, slice_dim=slice_dim, num_devices=num_devices)
    geometry, blocks = _plan(input_tensor, scenario)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(geometry["out_shape"])),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = create_program_descriptor(input_tensor, output_tensor, geometry, blocks)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
