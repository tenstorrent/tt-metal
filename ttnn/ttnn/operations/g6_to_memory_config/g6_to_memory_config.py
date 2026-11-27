# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_to_memory_config: return the input's tile pages, bit for bit, in the placement memory_config names.

One program per call: every page is read once from its source location and written once to its target
location by the core that holds one side of it (pull / push), or through an L1 staging slot when no core
holds either side (DRAM <-> DRAM bounce). See op_design.md.
"""

from __future__ import annotations

import math

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_memory_config_program_descriptor import (
    MODEL_L1_BUDGET_BYTES,
    buffer_name,
    create_program_descriptor,
    output_l1_bytes_per_core,
    scheme_name,
    select_plan,
    tile_grid,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS -- inputs is (scenario,)
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    return "tile_aligned" if shape[-2] % 32 == 0 and shape[-1] % 32 == 0 else "h_non_aligned"


def tag_in_scheme(inputs, axes):
    return inputs[0]["in_memory"]["scheme"]


def tag_in_buffer(inputs, axes):
    return inputs[0]["in_memory"]["buffer"]


def tag_out_scheme(inputs, axes):
    return inputs[0]["out_memory"]["scheme"]


def tag_out_buffer(inputs, axes):
    return inputs[0]["out_memory"]["buffer"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "rank": tag_rank,
    "alignment": tag_alignment,
    "in_scheme": tag_in_scheme,
    "in_buffer": tag_in_buffer,
    "out_scheme": tag_out_scheme,
    "out_buffer": tag_out_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

_SCHEMES = ["interleaved", "height_sharded", "width_sharded", "block_sharded"]

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "rank": [2, 3, 4],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "in_scheme": list(_SCHEMES),
    "in_buffer": ["dram", "l1"],
    "out_scheme": list(_SCHEMES),
    "out_buffer": ["dram", "l1"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # A target sharded over DRAM banks; only the bfloat4_b weights use it (block-format refinement).
    {"out_scheme": "width_sharded", "out_buffer": "dram"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _memory_entry(memory_config):
    return {"buffer": buffer_name(memory_config), "scheme": scheme_name(memory_config)}


def _check_shard_spec_cover(memory_config, rows, cols, what):
    """The shard spec covers the padded tile grid with one shard per core."""
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError(f"g6_to_memory_config: {what} is sharded but carries no shard spec")
    shard_h, shard_w = spec.shape[0], spec.shape[1]
    if shard_h <= 0 or shard_w <= 0 or shard_h % 32 or shard_w % 32:
        raise ValueError(f"g6_to_memory_config: {what} shard shape {tuple(spec.shape)} is not a multiple of 32")
    scheme = scheme_name(memory_config)
    if scheme == "height_sharded" and shard_w < cols * 32:
        raise ValueError(f"g6_to_memory_config: {what} height shard width {shard_w} < padded width {cols * 32}")
    if scheme == "width_sharded" and shard_h < rows * 32:
        raise ValueError(f"g6_to_memory_config: {what} width shard height {shard_h} < padded height {rows * 32}")
    n_rows, n_cols = math.ceil(rows / (shard_h // 32)), math.ceil(cols / (shard_w // 32))
    if scheme == "block_sharded":
        ranges = spec.grid.ranges()
        first = ranges[0]
        if len(ranges) != 1 or first.end.x - first.start.x + 1 < n_cols or first.end.y - first.start.y + 1 < n_rows:
            raise ValueError(
                f"g6_to_memory_config: {what} block shard grid must be one range of at least "
                f"{n_cols}x{n_rows} cores"
            )
    elif n_rows * n_cols > spec.grid.num_cores():
        raise ValueError(
            f"g6_to_memory_config: {what} needs {n_rows * n_cols} shards but its grid has "
            f"{spec.grid.num_cores()} cores"
        )


def validate(input_tensor, memory_config):
    # Structural refusals (ValueError) before the support contract.
    if not isinstance(input_tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_to_memory_config: input must be a tensor on device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_to_memory_config: input layout must be TILE, got {input_tensor.layout}")
    shape = list(input_tensor.shape)
    if not 2 <= len(shape) <= 4:
        raise ValueError(f"g6_to_memory_config: rank must be 2..4, got {len(shape)}")
    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError("g6_to_memory_config: memory_config must be a ttnn.MemoryConfig")
    in_config = input_tensor.memory_config()
    for config, what in ((in_config, "input"), (memory_config, "memory_config")):
        if config.is_sharded():
            if config.shard_spec is None:
                raise ValueError(f"g6_to_memory_config: {what} is sharded but carries no shard spec")
            if config.shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
                raise ValueError(f"g6_to_memory_config: {what} shard orientation must be ROW_MAJOR")
    rows, cols = tile_grid(shape)
    if in_config.is_sharded():
        _check_shard_spec_cover(in_config, rows, cols, "input")
    if memory_config.is_sharded():
        _check_shard_spec_cover(memory_config, rows, cols, "memory_config")

    scenario = {
        "shape": shape,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": _memory_entry(in_config),
        "out_memory": _memory_entry(memory_config),
    }
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_to_memory_config: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for cell in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in cell.items()):
            raise ExcludedCell(f"g6_to_memory_config: unsupported combination (refinement candidate): {cell}")

    # BB-05: the output's own L1 plus the op's buffers on the busiest core.
    plan = select_plan(input_tensor, memory_config)
    l1_bytes = output_l1_bytes_per_core(input_tensor, memory_config, plan) + plan.staging_bytes_per_core()
    if l1_bytes > MODEL_L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_to_memory_config: needs {l1_bytes} B of L1 per core, above the {MODEL_L1_BUDGET_BYTES} B budget"
        )
    return plan


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_to_memory_config(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    plan = validate(input_tensor, memory_config)
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    descriptor = create_program_descriptor(input_tensor, output_tensor, memory_config, plan)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
