# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_memory_config: the input's tile pages, bit for bit, placed as memory_config names.

    y[p] = x[p] for every tile page p; shape, dtype and TILE layout of the input; placement(y) = memory_config.

One device program per call; each page is read once from its source placement and written once to its
target placement through the L1 of the core that moves it.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_to_memory_config_program_descriptor import (
    check_shard_spec,
    dispatch,
    placement_of,
    tile_geometry,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# `inputs` is (scenario,): {"shape", "dtype", "layout", "in_memory": {"buffer", "scheme", ...},
# "out_memory": {...}} -- one production case of the golden suite, or the same dict validate()
# builds from a real call.


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

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "rank": [2, 3, 4],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "in_scheme": ["interleaved", "height_sharded", "width_sharded", "block_sharded"],
    "in_buffer": ["dram", "l1"],
    "out_scheme": ["interleaved", "height_sharded", "width_sharded", "block_sharded"],
    "out_buffer": ["dram", "l1"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # DRAM width-sharded targets are used only by the bfloat4_b weights; they arrive with the
    # block-format refinement.
    {"out_scheme": "width_sharded", "out_buffer": "dram"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_structure(input_tensor, memory_config):
    """Refusals the spec requires as ValueError; they precede the registry gate."""
    if not isinstance(input_tensor, ttnn.Tensor):
        raise ValueError("g6_to_memory_config: input_tensor must be a ttnn.Tensor")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_to_memory_config: input_tensor must be on a device")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_to_memory_config: input_tensor must be in TILE_LAYOUT, got {input_tensor.layout}")
    rank = len(input_tensor.shape)
    if rank not in (2, 3, 4):
        raise ValueError(f"g6_to_memory_config: rank must be 2 to 4, got {rank}")
    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError("g6_to_memory_config: memory_config must be a ttnn.MemoryConfig")
    geom = tile_geometry(input_tensor)
    src = placement_of(input_tensor.memory_config())
    dst = placement_of(memory_config)
    if src.is_sharded:
        check_shard_spec(src, geom, "input")
    if dst.is_sharded:
        check_shard_spec(dst, geom, "memory_config")
    return src, dst


def validate(input_tensor, memory_config):
    src, dst = _check_structure(input_tensor, memory_config)
    scenario = {
        "shape": [int(d) for d in input_tensor.shape],
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": {"buffer": src.buffer, "scheme": src.scheme},
        "out_memory": {"buffer": dst.buffer, "scheme": dst.scheme},
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

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
    return dispatch(input_tensor, memory_config)
