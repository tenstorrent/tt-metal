# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: broadcast x (d_0, d_1, 1, W) along dim 2 into a new (d_0, d_1, H_out, W) tensor, bit-exact.

One device program per call (regime `column_group_broadcast`, see op_design.md): each column group's
sender reads its x pages once, replicates row 0 of every tile in place and multicasts the expanded
tiles along its grid row; every core of the rectangle writes its own out rows of those columns.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import create_program_artifacts, select_plan


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _resolve_out_shape(shape, sizes):
    return [d if s == -1 else s for d, s in zip(shape, sizes)]


def tag_alignment(inputs, axes):
    """Tagged on the OUTPUT shape: sizes with every -1 replaced by x's size on that dim."""
    scenario = inputs[0]
    out_shape = _resolve_out_shape(scenario["shape"], scenario["sizes"])
    if out_shape[-1] % 32:
        return "w_non_aligned"
    if out_shape[-2] % 32:
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

_EXPANDED_DIM = 2


def validate(input_tensor, sizes):
    """Runtime gate. Returns the resolved output shape."""
    shape = list(input_tensor.shape)
    sizes = list(sizes)

    if len(shape) != 4 or len(sizes) != len(shape):
        raise ValueError(f"g6_expand: rank-4 input and 4 sizes required, got shape {shape}, sizes {sizes}")
    for s in sizes:
        if s < -1 or s == 0:
            raise ValueError(f"g6_expand: sizes entries must be -1 or >= 1, got {sizes}")
    for k, (d, s) in enumerate(zip(shape, sizes)):
        if d != 1 and s not in (-1, d):
            raise ValueError(f"g6_expand: sizes[{k}]={s} differs from the non-unit input size {d}")
        if k != _EXPANDED_DIM and d == 1 and s not in (-1, 1):
            raise ValueError(f"g6_expand: only dim {_EXPANDED_DIM} may be expanded, got sizes[{k}]={s}")
    if shape[_EXPANDED_DIM] != 1:
        raise ValueError(f"g6_expand: input must have one row per image (dim 2 == 1), got shape {shape}")

    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input_tensor must be on a device")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_expand: input_tensor must be DRAM interleaved, got {memory_config}")
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        tile = input_tensor.tile
        if list(tile.tile_shape) != [32, 32] or list(tile.face_shape) != [16, 16]:
            raise ValueError(
                f"g6_expand: 32x32 tiles with 16x16 faces required, got {tile.tile_shape} / {tile.face_shape}"
            )

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": shape, "sizes": sizes}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")

    return _resolve_out_shape(shape, sizes)


# ---------------------------------------------------------------------------
# Program artifacts, cached per key (BB-07)
# ---------------------------------------------------------------------------

_ARTIFACT_CACHE = {}


def _cache_key(input_tensor, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        tuple(input_tensor.padded_shape),
        tuple(input_tensor.shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        device.id(),
        (grid.x, grid.y),
    )


def _artifacts(input_tensor, output_tensor, out_shape):
    key = _cache_key(input_tensor, out_shape)
    artifacts = _ARTIFACT_CACHE.get(key)
    if artifacts is None:
        artifacts = create_program_artifacts(input_tensor, output_tensor)
        _ARTIFACT_CACHE[key] = artifacts
    return artifacts


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    out_shape = validate(input_tensor, sizes)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    artifacts = _artifacts(input_tensor, output_tensor, out_shape)
    descriptor = artifacts.bind(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def g6_expand_plan(input_tensor: ttnn.Tensor, sizes) -> dict:
    """The plan a call with these arguments selects (BB-08)."""
    out_shape = validate(input_tensor, sizes)
    plan = select_plan(input_tensor, out_shape)
    return {
        "regime": plan.regime,
        "num_col_groups": plan.num_col_groups,
        "num_row_groups": plan.num_row_groups,
        "core_src_col_tiles": plan.core_src_col_tiles,
        "core_out_row_tiles": plan.core_out_row_tiles,
        "block_src_col_tiles": plan.block_src_col_tiles,
        "block_out_row_tiles": plan.block_out_row_tiles,
        "src_block_depth": plan.src_block_depth,
        "seed_rows": plan.seed_rows,
        "num_cores": plan.num_cores,
        "l1_bytes_per_core": plan.l1_bytes_per_core,
        "core_roles": plan.core_roles(),
    }
