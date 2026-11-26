# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand — materialized expand of x (d0, d1, 1, W) along dim 2 into shape `sizes`.

out[n0, n1, h, w] = x[n0, n1, 0, w]; bit-exact; one device program per call.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import (
    BCAST_BUFFER_DEPTH,
    BLOCK_IMAGES,
    MIN_BLOCK_TILE_ROWS,
    compute_plan,
    create_program_descriptor,
    patch_addresses,
)

TILE_HW = 32


def _out_shape(shape, sizes):
    return [int(shape[d]) if int(s) == -1 else int(s) for d, s in enumerate(sizes)]


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Alignment of the OUTPUT shape (sizes with -1 replaced by x's size)."""
    scenario = inputs[0]
    out = _out_shape(scenario["shape"], scenario["sizes"])
    w_out, h_out = out[-1], out[-2]
    if w_out % TILE_HW != 0:
        return "w_non_aligned"
    if h_out % TILE_HW != 0:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}

# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned"],
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


def _validate_structure(input_tensor, sizes):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input_tensor must be on device")
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_expand: input rank must be 4, got {len(shape)}")
    try:
        sizes = [int(s) for s in sizes]
    except TypeError as e:
        raise ValueError(f"g6_expand: sizes must be a sequence of ints, got {sizes!r}") from e
    if len(sizes) != len(shape):
        raise ValueError(f"g6_expand: len(sizes)={len(sizes)} differs from input rank {len(shape)}")
    for d, s in enumerate(sizes):
        if s == 0 or (s < 0 and s != -1):
            raise ValueError(f"g6_expand: sizes[{d}]={s} is invalid (must be -1 or > 0)")
        if shape[d] != 1 and s not in (-1, shape[d]):
            raise ValueError(f"g6_expand: sizes[{d}]={s} differs from input size {shape[d]} (not a size-1 dim)")
        if d != 2 and shape[d] == 1 and s not in (-1, 1):
            raise ValueError(f"g6_expand: expansion of dim {d} is outside this contract (only dim 2)")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: input dim 2 must be 1, got {shape[2]}")
    mc = input_tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"g6_expand: input must be DRAM interleaved, got {mc}")
    return shape, sizes


def validate(input_tensor, sizes):
    shape, sizes = _validate_structure(input_tensor, sizes)
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
    scenario = {"shape": shape, "sizes": sizes}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")
    return shape, sizes


# ---------------------------------------------------------------------------
# Plan / program cache (BB-07, BB-08)
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _plan_for(input_tensor, shape, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return compute_plan(
        shape,
        out_shape,
        grid.x,
        grid.y,
        input_tensor.buffer_page_size(),
        input_tensor.element_size(),
        block_images=BLOCK_IMAGES,
        min_block_tile_rows=MIN_BLOCK_TILE_ROWS,
        bcast_buffer_depth=BCAST_BUFFER_DEPTH,
    )


def _cache_key(input_tensor, shape, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        device.id(),
        (grid.x, grid.y),
        BLOCK_IMAGES,
        MIN_BLOCK_TILE_ROWS,
        BCAST_BUFFER_DEPTH,
    )


def describe_plan(input_tensor, sizes):
    """Plan a call selects (BB-08) and the per-core role/block range dump (BB-21)."""
    shape, sizes = validate(input_tensor, sizes)
    out_shape = _out_shape(shape, sizes)
    plan = _plan_for(input_tensor, shape, out_shape)
    report = {k: v for k, v in plan.items() if k != "core_blocks"}
    report["cores"] = [
        {"core": xy, "role": "reader+writer", "first_block": fb, "num_blocks": nb} for xy, fb, nb in plan["core_blocks"]
    ]
    return report


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    shape, sizes = validate(input_tensor, sizes)
    out_shape = _out_shape(shape, sizes)
    device = input_tensor.device()

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, shape, out_shape)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        plan = _plan_for(input_tensor, shape, out_shape)
        descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = (plan, descriptor)
    else:
        plan, descriptor = entry
        patch_addresses(descriptor, plan, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
