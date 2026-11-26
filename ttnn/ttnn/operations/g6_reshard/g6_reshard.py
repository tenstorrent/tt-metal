# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard — move a BLOCK_SHARDED L1 tensor, tile for tile as stored, into a new
BLOCK_SHARDED L1 tensor on the placement `memory_config` names. One device program."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import (
    TILE,
    Placement,
    create_program_descriptor,
    describe_plan as _describe_plan,
    make_plan,
    patch_addresses,
    tensor_tiles,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — no shape-derived axis: the placements are arguments, not axes.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = {}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, memory_config):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((input_tensor.shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_reshard: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_reshard: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Argument checks (ValueError / RuntimeError) — not support refusals
# ---------------------------------------------------------------------------

# L1 the model leaves free per core; the op's only allocation is out's shard.
L1_FREE_BYTES = 1347776


def _is_block_sharded_l1(memory_config) -> bool:
    return (
        memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        and memory_config.buffer_type == ttnn.BufferType.L1
        and memory_config.shard_spec is not None
    )


def _check_input(input_tensor) -> None:
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshard: input_tensor must be on a device")
    if not _is_block_sharded_l1(input_tensor.memory_config()):
        raise ValueError("g6_reshard: input_tensor must be BLOCK_SHARDED in L1 with a shard spec")
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_reshard: input_tensor must have rank 4, got {len(shape)}")
    height = 1
    for d in shape[:-1]:
        height *= d
    if height % TILE or shape[-1] % TILE:
        raise ValueError(f"g6_reshard: 2-D view ({height}, {shape[-1]}) must have H and W multiples of {TILE}")


def _check_memory_config(memory_config, device, tensor_rows_tiles: int, tensor_cols_tiles: int):
    """Returns memory_config with its grid as a single CoreRange (the 2-D shard distribution
    accepts only one range; a rectangle given as several ranges is the same placement)."""
    if not isinstance(memory_config, ttnn.MemoryConfig) or not _is_block_sharded_l1(memory_config):
        raise ValueError("g6_reshard: memory_config must be BLOCK_SHARDED in L1 with a shard spec")
    spec = memory_config.shard_spec
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    if shard_h <= 0 or shard_w <= 0 or shard_h % TILE or shard_w % TILE:
        raise ValueError(f"g6_reshard: memory_config shard shape ({shard_h}, {shard_w}) must be multiples of {TILE}")
    box = spec.grid.bounding_box()
    if (box.end.x - box.start.x + 1) * (box.end.y - box.start.y + 1) != spec.grid.num_cores():
        raise ValueError("g6_reshard: memory_config core grid must be one rectangle")
    worker = device.compute_with_storage_grid_size()
    if box.end.x >= worker.x or box.end.y >= worker.y:
        raise ValueError(
            f"g6_reshard: memory_config grid ends at ({box.end.x}, {box.end.y}), "
            f"outside the {worker.x}x{worker.y} worker grid"
        )
    p_out = Placement.from_memory_config(memory_config)
    need_x, need_y = p_out.trimmed_extent(
        p_out.num_block_rows(tensor_rows_tiles), p_out.num_block_cols(tensor_cols_tiles)
    )
    have_x, have_y = box.end.x - box.start.x + 1, box.end.y - box.start.y + 1
    if need_x > have_x or need_y > have_y:
        raise ValueError(
            f"g6_reshard: memory_config grid {have_x}x{have_y} cores cannot hold the "
            f"{need_x}x{need_y} block grid of out in its orientation"
        )
    if len(spec.grid.ranges()) != 1:
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ttnn.CoreRangeSet([box]), [shard_h, shard_w], spec.orientation),
        )
    return memory_config


# ---------------------------------------------------------------------------
# Program cache (BB-07): the descriptor is a function of the plan only; buffer addresses are
# the only per-call values and are patched into common RT args 0, 1 on a hit.
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _plan_for(input_tensor, memory_config):
    page_bytes = input_tensor.buffer_aligned_page_size()
    if page_bytes % ttnn.get_l1_alignment():
        raise RuntimeError(f"g6_reshard: page size {page_bytes} B is not L1-aligned")
    return make_plan(
        input_tensor.padded_shape,
        page_bytes,
        Placement.from_memory_config(input_tensor.memory_config()),
        Placement.from_memory_config(memory_config),
    )


def describe_plan(input_tensor, memory_config) -> dict:
    """Which regime, issuers and runs a call would use, per core. No dispatch."""
    return _describe_plan(_plan_for(input_tensor, memory_config))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    _check_input(input_tensor)

    device = input_tensor.device()
    memory_config = _check_memory_config(memory_config, device, *tensor_tiles(input_tensor.padded_shape))
    plan = _plan_for(input_tensor, memory_config)

    out_shard_bytes = plan.p_out.shard_rows_tiles * plan.p_out.shard_cols_tiles * plan.page_bytes
    if out_shard_bytes > L1_FREE_BYTES:
        raise RuntimeError(f"g6_reshard: out shard of {out_shard_bytes} B exceeds the {L1_FREE_BYTES} B L1 budget")

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        memory_config,
    )
    if out.buffer_aligned_page_size() != plan.page_bytes:
        raise RuntimeError("g6_reshard: in0 and out page sizes differ")

    in_addr, out_addr = input_tensor.buffer_address(), out.buffer_address()
    key = (device.id(), plan)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(device, plan, in_addr, out_addr)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        patch_addresses(descriptor, in_addr, out_addr)

    return ttnn.generic_op([input_tensor, out], descriptor)
