# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_memory_config: move a TILE tensor's pages into the placement memory_config names.

    y[p] = x[p] for every tile page p (bit copy); shape, dtype, TILE layout kept;
    placement(y) = memory_config.

One device program per call (ttnn.generic_op); see op_design.md for the regimes.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_memory_config_program_descriptor import (
    L1_BUDGET_BYTES,
    Plan,
    create_program_descriptor,
    dump_core_roles,
    geometry_of,
    op_l1_bytes_per_core,
    placement_of,
    select_plan,
    shard_layout_error,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), the scenario dict of one production case
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    if shape[-2] % 32 == 0 and shape[-1] % 32 == 0:
        return "tile_aligned"
    return "h_non_aligned"


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
    # DRAM width-sharded target: used only by the bfloat4_b weights; leaves with the block-format refinement.
    {"out_scheme": "width_sharded", "out_buffer": "dram"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _structural_error(input_tensor, memory_config):
    """Refusals the spec requires as ValueError; they precede the SUPPORTED gate."""
    if not isinstance(input_tensor, ttnn.Tensor):
        return "input_tensor must be a ttnn.Tensor"
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        return "input_tensor must be on a device"
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return f"input_tensor must be in TILE_LAYOUT, got {input_tensor.layout}"
    rank = len(input_tensor.shape)
    if rank not in (2, 3, 4):
        return f"rank must be 2, 3 or 4, got {rank}"
    if not isinstance(memory_config, ttnn.MemoryConfig):
        return "memory_config must be a ttnn.MemoryConfig"
    geom = geometry_of(input_tensor)
    for name, config in (("input_tensor", input_tensor.memory_config()), ("memory_config", memory_config)):
        placement = placement_of(config)
        if placement.is_sharded:
            reason = shard_layout_error(placement, geom)
            if reason is not None:
                return f"{name} shard spec: {reason}"
    return None


def _axes(input_tensor, memory_config):
    in_placement = placement_of(input_tensor.memory_config())
    out_placement = placement_of(memory_config)
    scenario = {
        "shape": list(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": {"buffer": in_placement.buffer, "scheme": in_placement.scheme},
        "out_memory": {"buffer": out_placement.buffer, "scheme": out_placement.scheme},
    }
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)
    return axes


def validate(input_tensor, memory_config):
    reason = _structural_error(input_tensor, memory_config)
    if reason is not None:
        raise ValueError(f"g6_to_memory_config: {reason}")

    axes = _axes(input_tensor, memory_config)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_to_memory_config: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_to_memory_config: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_to_memory_config(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)

    l1_bytes = op_l1_bytes_per_core(input_tensor, memory_config)
    if l1_bytes > L1_BUDGET_BYTES:
        raise ValueError(
            f"g6_to_memory_config: needs {l1_bytes} B of L1 on a core (output + op buffers), "
            f"over the {L1_BUDGET_BYTES} B the model leaves to the op"
        )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )
    descriptor = create_program_descriptor(input_tensor, output_tensor, memory_config)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


__all__ = [
    "EXCLUSIONS",
    "INPUT_TAGGERS",
    "Plan",
    "SUPPORTED",
    "dump_core_roles",
    "g6_to_memory_config",
    "select_plan",
    "validate",
]
