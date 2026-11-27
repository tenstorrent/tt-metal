# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: broadcast x (d_0, d_1, 1, W) along dim 2 to (d_0, d_1, H_out, W), bit-exact, one program.

Regime `column_group_broadcast` (op_design.md): column groups along grid y own contiguous x page
ranges; each group's sender (x = 0) reads its pages once, expands row 0 into whole tiles in L1 and
multicasts them along its grid row; every core of the group writes its own out tile rows.
"""

from __future__ import annotations

import math

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _resolve_out_shape(shape, sizes):
    return [int(d) if int(s) == -1 else int(s) for d, s in zip(shape, sizes)]


def tag_alignment(inputs, axes):
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


def validate(input_tensor, sizes):
    shape = list(input_tensor.shape)
    sizes = list(sizes)
    if len(shape) != 4 or len(sizes) != len(shape):
        raise ValueError(f"g6_expand: need a rank-4 input and 4 sizes, got shape {shape} and sizes {sizes}")
    for s in sizes:
        if int(s) < -1 or int(s) == 0:
            raise ValueError(f"g6_expand: every size must be -1 or >= 1, got {sizes}")
    for k, (d, s) in enumerate(zip(shape, sizes)):
        s = int(s)
        if d != 1 and s not in (-1, d):
            raise ValueError(f"g6_expand: dim {k} has size {d} != 1 and cannot become {s}")
        if k != 2 and d == 1 and s not in (-1, 1):
            raise ValueError(f"g6_expand: only dim 2 may be expanded, dim {k} asks for {s}")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: x must have one row per image (dim 2 == 1), got {shape}")
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
                f"g6_expand: needs 32x32 tiles with 16x16 faces, got {tile.tile_shape} / {tile.face_shape}"
            )

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(({"shape": shape, "sizes": sizes},), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# select_plan — the single source of every knob
# ---------------------------------------------------------------------------

REGIME = "column_group_broadcast"
# L1 the model leaves to the op on every core (BB-05).
L1_OP_BUDGET = 1347776
# Rows of each upper face the RISC-V seeds before NoC self-reads replicate them; must divide face height.
SEED_ROWS = 2


def select_plan(input_tensor, out_shape):
    shape = list(input_tensor.shape)
    tile_h, tile_w = (int(v) for v in input_tensor.tile.tile_shape)
    face_h, face_w = (int(v) for v in input_tensor.tile.face_shape)
    page_bytes = int(input_tensor.buffer_page_size())

    tensor_w_tiles = math.ceil(shape[3] / tile_w)
    tensor_src_col_tiles = shape[0] * shape[1] * tensor_w_tiles
    tensor_out_row_tiles = math.ceil(out_shape[2] / tile_h)

    grid = input_tensor.device().compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    max_col_groups = grid_y
    max_row_groups = grid_x

    core_src_col_tiles = math.ceil(tensor_src_col_tiles / min(tensor_src_col_tiles, max_col_groups))
    num_col_groups = math.ceil(tensor_src_col_tiles / core_src_col_tiles)
    core_out_row_tiles = math.ceil(tensor_out_row_tiles / min(tensor_out_row_tiles, max_row_groups))
    num_row_groups = math.ceil(tensor_out_row_tiles / core_out_row_tiles)

    src_block_depth = 1 if core_src_col_tiles * page_bytes <= L1_OP_BUDGET else 2
    block_src_col_tiles = min(core_src_col_tiles, L1_OP_BUDGET // (src_block_depth * page_bytes))
    block_out_row_tiles = core_out_row_tiles

    face_rows = face_h
    assert face_rows % SEED_ROWS == 0 and SEED_ROWS >= 1, "seed_rows must divide the face height"

    return {
        "regime": REGIME,
        "page_bytes": page_bytes,
        "tensor_w_tiles": tensor_w_tiles,
        "tensor_src_col_tiles": tensor_src_col_tiles,
        "tensor_out_row_tiles": tensor_out_row_tiles,
        "grid": (grid_x, grid_y),
        "num_col_groups": num_col_groups,
        "num_row_groups": num_row_groups,
        "core_src_col_tiles": core_src_col_tiles,
        "core_out_row_tiles": core_out_row_tiles,
        "block_src_col_tiles": block_src_col_tiles,
        "block_out_row_tiles": block_out_row_tiles,
        "src_block_depth": src_block_depth,
        "seed_rows": SEED_ROWS,
        "face_rows": face_rows,
        "face_row_bytes": face_w * input_tensor.element_size(),
        "face_bytes": face_h * face_w * input_tensor.element_size(),
        "faces_per_tile_row": tile_w // face_w,
        "face_rows_per_tile": tile_h // face_h,
        "num_cores": num_col_groups * num_row_groups,
        "l1_bytes_per_core": src_block_depth * block_src_col_tiles * page_bytes,
        "core_roles": {
            (x, y): ("sender" if x == 0 else "receiver") for y in range(num_col_groups) for x in range(num_row_groups)
        },
    }


_PLAN_REPORT_KEYS = (
    "regime",
    "num_col_groups",
    "num_row_groups",
    "core_src_col_tiles",
    "core_out_row_tiles",
    "block_src_col_tiles",
    "block_out_row_tiles",
    "src_block_depth",
    "seed_rows",
    "num_cores",
    "l1_bytes_per_core",
    "core_roles",
)


def g6_expand_plan(input_tensor, sizes):
    """The plan the entry point selects for this call (BB-08)."""
    validate(input_tensor, sizes)
    plan = select_plan(input_tensor, _resolve_out_shape(list(input_tensor.shape), list(sizes)))
    return {k: plan[k] for k in _PLAN_REPORT_KEYS}


# ---------------------------------------------------------------------------
# Program cache (BB-07): a hit only rebinds the two buffer addresses.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _cache_key(input_tensor, out_shape):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        tuple(input_tensor.padded_shape),
        tuple(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        device.id(),
        (int(grid.x), int(grid.y)),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_expand(input_tensor, sizes):
    validate(input_tensor, sizes)
    out_shape = _resolve_out_shape(list(input_tensor.shape), list(sizes))
    device = input_tensor.device()

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, out_shape)
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = create_program_descriptor(input_tensor, output_tensor, select_plan(input_tensor, out_shape))
        _PROGRAM_CACHE[key] = program

    descriptor = program.bind(input_tensor.buffer_address(), output_tensor.buffer_address())
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
