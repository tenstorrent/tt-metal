# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_pad: constant high-end padding of the trailing k dims of a rank-2/3/4 tensor.

One device program per call (op_design.md):
  extension_band    -- TILE, only dim -2 grows inside the existing last tile row: c is written into the
                       band rows of the last tile row of every slice IN PLACE, and the output is a
                       metadata-only view of the input's storage (no program, no data moved).
  new_storage_tiles -- every other TILE call: each output page is assembled once in L1 (input tile + fill
                       patches, or a pure fill page) and written once.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_pad_program_descriptor import (
    RT_ADDR,
    RT_FILL_BITS,
    build_extension_descriptor,
    build_new_storage_descriptor,
)

TILE = 32
FACE = 16
L1_BUDGET_BYTES = 1347776  # BB-05: L1 the model leaves to the op on every core


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs = (s,), s = scenario dict)
# ---------------------------------------------------------------------------


def _highs(shape, padding):
    shape = list(shape)
    pairs = [tuple(p) for p in padding]
    return [0] * (len(shape) - len(pairs)) + [int(p[1]) for p in pairs]


def tag_alignment(inputs, axes):
    s = inputs[0]
    H, W = s["shape"][-2], s["shape"][-1]
    if W % TILE != 0:
        return "w_non_aligned"
    if H % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0]["shape"])


def tag_padded_dims(inputs, axes):
    s = inputs[0]
    h = _highs(s["shape"], s["padding"])
    hh, hw = h[-2] > 0, h[-1] > 0
    if hh and hw:
        return "hw"
    if hh:
        return "h"
    if hw:
        return "w"
    return "leading"


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_memory_config(inputs, axes):
    return inputs[0]["memory_config"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "memory_config": tag_memory_config,
    "alignment": tag_alignment,
    "rank": tag_rank,
    "padded_dims": tag_padded_dims,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["dram"],
    "alignment": ["w_non_aligned", "h_non_aligned"],
    "rank": [3, 4],
    "padded_dims": ["h", "hw"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Block knobs (tunables; each defined once here, every dependent quantity derives from them)
# ---------------------------------------------------------------------------

KNOBS = {
    # Fewest work units (band tiles / output tiles) a core gets: ncores = min(grid, ceil(N / this)).
    "min_units_per_core": 1,
    # extension_band: number of DM RISC lanes the per-core band range is split across (1 or 2).
    "band_lanes": 1,
    # extension_band: band tiles per block (0 = whole per-lane range, the design's Phase 0 value).
    "band_block_tiles": 0,
    # new_storage_tiles: output tiles per block (0 = whole per-core range, clamped by the L1 cap).
    "block_tiles": 0,
    # new_storage_tiles: cb_input_tiles depth in blocks (clamped to the number of blocks per core).
    "input_depth_blocks": 2,
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _int_range(dtype):
    if dtype == ttnn.int32:
        return (-(2**31), 2**31 - 1)
    if dtype == ttnn.uint32:
        return (0, 2**32 - 1)
    return None


def _check_arguments(input_tensor, padding, value):
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError("g6_pad: input_tensor must be a ttnn.Tensor")
    r = len(input_tensor.shape)
    if r < 2 or r > 4:
        raise ValueError(f"g6_pad: rank must be 2..4, got {r}")
    if not isinstance(padding, (list, tuple)):
        raise ValueError("g6_pad: padding must be a list or tuple of (low, high) pairs")
    k = len(padding)
    if k < 1 or k > r:
        raise ValueError(f"g6_pad: padding must hold 1..{r} pairs, got {k}")
    for p in padding:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"g6_pad: each padding entry must be a (low, high) pair, got {p!r}")
        for v in p:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"g6_pad: padding values must be ints, got {p!r}")
        if p[0] != 0:
            raise ValueError(f"g6_pad: only low = 0 is supported, got {p!r}")
        if p[1] < 0:
            raise ValueError(f"g6_pad: high must be >= 0, got {p!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"g6_pad: value must be an int or a float, got {type(value).__name__}")
    rng = _int_range(input_tensor.dtype)
    if rng is not None:
        if not isinstance(value, int) and not float(value).is_integer():
            raise ValueError(f"g6_pad: value {value!r} is not an integer for dtype {input_tensor.dtype}")
        if not (rng[0] <= int(value) <= rng[1]):
            raise ValueError(f"g6_pad: value {value!r} out of range for dtype {input_tensor.dtype}")
    if input_tensor.is_sharded():
        raise ValueError("g6_pad: sharded input_tensor is not supported")


def _memory_tag(input_tensor):
    return "l1" if input_tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def validate(input_tensor, padding, value=0.0):
    # (1) argument validation -> ValueError
    _check_arguments(input_tensor, padding, value)

    # (2) scenario dict from the tensor, then SUPPORTED per axis, then EXCLUSIONS
    s = {
        "shape": list(input_tensor.shape),
        "padding": [list(p) for p in padding],
        "value": value,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_config": _memory_tag(input_tensor),
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((s,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_pad: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_pad: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Plan (single source of truth for every block knob and derived geometry)
# ---------------------------------------------------------------------------


def fill_bits_of(value, dtype):
    """32-bit pattern of c = value converted to dtype as torch converts it (bf16: RNE of fp32)."""
    import torch

    if dtype == ttnn.bfloat16:
        b = int(torch.tensor(value, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
        return (b << 16) | b
    if dtype == ttnn.float32:
        return int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    return int(value) & 0xFFFFFFFF


def select_regime(layout, in_shape, out_shape):
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return "new_storage_rows"
    highs = [D - d for D, d in zip(out_shape, in_shape)]
    r = len(highs)
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != r - 2)
    same_tile_rows = math.ceil(in_shape[-2] / TILE) == math.ceil(out_shape[-2] / TILE)
    return "extension_band" if (only_h and same_tile_rows) else "new_storage_tiles"


def _lead2(shape):
    lead = [1, 1] + list(shape[:-2])
    return (lead[-2], lead[-1])


def _band_segments(row_begin, row_end, row_bytes, face_bytes):
    """Coalesced (offset, bytes) runs covering rows [row_begin, row_end) of one tile, all four faces."""
    runs = []
    top = (row_begin, min(row_end, FACE))
    bot = (max(row_begin, FACE) - FACE, row_end - FACE)
    for face, (a, b) in ((0, top), (1, top), (2, bot), (3, bot)):
        if b > a:
            runs.append([face * face_bytes + a * row_bytes, (b - a) * row_bytes])
    merged = []
    for off, ln in runs:
        if merged and merged[-1][0] + merged[-1][1] == off:
            merged[-1][1] += ln
        else:
            merged.append([off, ln])
    return [tuple(m) for m in merged]


def make_plan(input_tensor, out_shape, value, knobs):
    in_shape = list(input_tensor.shape)
    in_padded = list(input_tensor.padded_shape)
    dtype = input_tensor.dtype
    elem_bytes = input_tensor.element_size()
    row_bytes = FACE * elem_bytes
    face_bytes = FACE * row_bytes
    tile_bytes = 4 * face_bytes
    regime = select_regime(input_tensor.layout, in_shape, out_shape)
    grid = input_tensor.device().compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y

    in_tiles_h = in_padded[-2] // TILE
    in_tiles_w = in_padded[-1] // TILE
    in_lead = _lead2(in_shape)
    plan = {
        "regime": regime,
        "dtype": dtype,
        "elem_bytes": elem_bytes,
        "tile_bytes": tile_bytes,
        "in_tiles_h": in_tiles_h,
        "in_tiles_w": in_tiles_w,
        "in_lead": in_lead,
        "fill_bits": fill_bits_of(value, dtype),
        "min_units_per_core": knobs["min_units_per_core"],
    }
    if regime == "extension_band":
        row_begin = in_shape[-2] % TILE
        row_end = out_shape[-2] - TILE * (in_tiles_h - 1)
        assert 1 <= row_begin < row_end <= TILE, (row_begin, row_end)
        plan["n_units"] = in_lead[0] * in_lead[1] * in_tiles_w
        plan["band_segments"] = _band_segments(row_begin, row_end, row_bytes, face_bytes)
        plan["band_lanes"] = knobs["band_lanes"]
        plan["band_block_tiles"] = knobs["band_block_tiles"]
    elif regime == "new_storage_tiles":
        out_lead = _lead2(out_shape)
        out_tiles_h = -(-out_shape[-2] // TILE)
        out_tiles_w = -(-out_shape[-1] // TILE)
        plan["out_lead"] = out_lead
        plan["out_tiles_h"] = out_tiles_h
        plan["out_tiles_w"] = out_tiles_w
        plan["n_units"] = out_lead[0] * out_lead[1] * out_tiles_h * out_tiles_w
        plan["edge_row"] = in_shape[-2] % TILE
        plan["edge_col"] = in_shape[-1] % TILE
        plan["block_tiles"] = knobs["block_tiles"]
        plan["input_depth_blocks"] = knobs["input_depth_blocks"]
        # L1 cap (BB-05): fill page + depth-2 blocks must fit the budget.
        plan["block_tiles_cap"] = (L1_BUDGET_BYTES - tile_bytes) // (2 * tile_bytes)
    else:
        raise UnsupportedAxisValue(f"g6_pad: regime {regime} is not built")
    plan["ncores"] = min(grid_cores, -(-plan["n_units"] // max(1, knobs["min_units_per_core"])))
    return plan


# ---------------------------------------------------------------------------
# Program cache (BB-07): structure keyed by shape; per-call values patched into RT args on a hit.
# ---------------------------------------------------------------------------

_CACHE = {}
last_plan = None


def _cache_key(regime, input_tensor, out_shape, knobs):
    grid = input_tensor.device().compute_with_storage_grid_size()
    return (
        regime,
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.memory_config().buffer_type,
        (grid.x, grid.y),
        tuple(sorted(knobs.items())),
    )


def _patch(desc, patches, addr, fill_bits):
    for k, x, y in patches:
        args = desc.kernels[k].runtime_args[x][y]
        args[RT_ADDR] = addr
        if fill_bits is not None:
            args[RT_FILL_BITS] = fill_bits


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_pad(input_tensor, padding, value=0.0):
    validate(input_tensor, padding, value)
    global last_plan

    in_shape = list(input_tensor.shape)
    out_shape = [d + h for d, h in zip(in_shape, _highs(in_shape, padding))]
    knobs = dict(KNOBS)
    regime = select_regime(input_tensor.layout, in_shape, out_shape)
    key = _cache_key(regime, input_tensor, out_shape, knobs)
    fill_bits = fill_bits_of(value, input_tensor.dtype)

    if regime == "extension_band":
        # Metadata-only view of the input's storage at the output logical shape, same padded shape:
        # verified under ttnn.graph capture to add zero program nodes and keep the buffer address.
        output = ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)
        entry = _CACHE.get(key)
        if entry is None:
            plan = make_plan(input_tensor, out_shape, value, knobs)
            desc, patches = build_extension_descriptor(plan, output)
            entry = _CACHE[key] = (plan, desc, patches)
        plan, desc, patches = entry
        _patch(desc, patches, output.buffer_address(), fill_bits)
        last_plan = plan
        return ttnn.generic_op([output], desc)

    # new_storage_tiles
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    entry = _CACHE.get(key)
    if entry is None:
        plan = make_plan(input_tensor, out_shape, value, knobs)
        desc, patches = build_new_storage_descriptor(plan, input_tensor, output)
        entry = _CACHE[key] = (plan, desc, patches)
    plan, desc, (patches_in, patches_out) = entry
    _patch(desc, patches_in, input_tensor.buffer_address(), None)
    _patch(desc, patches_out, output.buffer_address(), fill_bits)
    last_plan = plan
    return ttnn.generic_op([input_tensor, output], desc)
