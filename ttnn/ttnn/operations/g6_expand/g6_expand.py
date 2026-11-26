# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: x (d_0, d_1, 1, W) -> x.expand(*sizes).contiguous(), bit-exact, one device program.

Regime R1 (column_mcast_2d, op_design.md): the pattern axis (input pages) rides grid columns, the
replica axis (output tile rows) rides grid rows. The top core of each column fetches row 0 of its
input pages, replicates it into full pattern tiles and multicasts them down the column; every core
writes its replica rows of its pattern pages.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import Plan, create_program_descriptor, patch_buffer_addresses, select_plan

TILE = 32


def output_shape(shape, sizes):
    """sizes with every -1 replaced by the input's size along that dim."""
    return [int(d) if int(s) == -1 else int(s) for d, s in zip(shape, sizes)]


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Alignment of the OUTPUT shape: W_out decides first, then H_out."""
    scenario = inputs[0]
    out = output_shape(scenario["shape"], scenario["sizes"])
    if out[-1] % TILE:
        return "w_non_aligned"
    if out[-2] % TILE:
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


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _validate_structure(input_tensor, sizes):
    """Contract checks the taggers rely on (a well-formed output shape); ValueError on violation."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError(f"g6_expand: input_tensor must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input_tensor must be on device")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_expand: input_tensor must be DRAM interleaved, got {memory_config}")
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_expand: input_tensor must have rank 4, got shape {shape}")
    if not isinstance(sizes, (list, tuple)) or len(sizes) != 4:
        raise ValueError(f"g6_expand: sizes must be a list or tuple of 4 ints, got {sizes!r}")
    for s in sizes:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"g6_expand: sizes must hold ints, got {sizes!r}")
        if s != -1 and s < 1:
            raise ValueError(f"g6_expand: every size must be -1 or >= 1, got {sizes!r}")
    for dim, (d, s) in enumerate(zip(shape, sizes)):
        if s == -1 or s == d:
            continue
        if d != 1:
            raise ValueError(f"g6_expand: size {s} of dim {dim} differs from the input's non-1 size {d}")
        if dim != 2:
            raise ValueError(f"g6_expand: only dim 2 may be expanded, got sizes {list(sizes)} for shape {shape}")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: input dim 2 must be 1, got shape {shape}")


def _axes(input_tensor, sizes):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": list(input_tensor.shape), "sizes": list(sizes)}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)
    return axes


def validate(input_tensor, sizes):
    _validate_structure(input_tensor, sizes)
    axes = _axes(input_tensor, sizes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# One cached descriptor per program shape; a hit patches only the two buffer addresses.
_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor, out_shape, device, grid):
    memory_config = input_tensor.memory_config()
    return (
        tuple(input_tensor.shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        device.id(),
        (grid.x, grid.y),
    )


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    """Materialize input_tensor.expand(*sizes) as a new tensor in the input's placement."""
    validate(input_tensor, sizes)

    device = input_tensor.device()
    out_shape = output_shape(input_tensor.shape, sizes)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    grid = device.compute_with_storage_grid_size()
    key = _cache_key(input_tensor, out_shape, device, grid)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        plan = select_plan(
            list(input_tensor.shape),
            out_shape,
            (grid.x, grid.y),
            tile_bytes=input_tensor.buffer_aligned_page_size(),
        )
        entry = create_program_descriptor(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = entry
    else:
        patch_buffer_addresses(entry, input_tensor, output_tensor)
    g6_expand.last_plan = entry.plan

    return ttnn.generic_op([input_tensor, output_tensor], entry.descriptor)


g6_expand.last_plan = None

__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "Plan",
    "SUPPORTED",
    "g6_expand",
    "output_shape",
    "select_plan",
    "tag_alignment",
    "validate",
]
