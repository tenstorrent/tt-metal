# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a new tensor
BLOCK_SHARDED in L1 on the placement memory_config names. One device program (regime PULL, see
g6_reshard_program_descriptor.py); nothing is computed or converted.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import (
    NUM_ISSUERS,
    TILE_DIM,
    ReshardPlan,
    create_program_descriptor,
    patch_addresses,
    placement_from_shard_spec,
    tile_grid,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Placements are arguments of every call, never axes; the spec names no shape-derived axis.
INPUT_TAGGERS = {}

SUPPORTED = {
    "dtype": [ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
}

EXCLUSIONS = []


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
# Host validation and planning
# ---------------------------------------------------------------------------

# L1 per core the model leaves to the op (BB-05); out's shard is the op's only allocation.
L1_BUDGET_BYTES = 1347776


def _is_one_rectangle(grid) -> bool:
    box = grid.bounding_box()
    area = (box.end.x - box.start.x + 1) * (box.end.y - box.start.y + 1)
    return grid.num_cores() == area


def _check_block_sharded_l1(memory_config, what: str) -> None:
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise ValueError(f"g6_reshard: {what} must be BLOCK_SHARDED, got {memory_config.memory_layout}")
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_reshard: {what} must be in L1, got {memory_config.buffer_type}")
    if memory_config.shard_spec is None:
        raise ValueError(f"g6_reshard: {what} has no shard spec")


def _check_2d_view_matches_buffer(images: int, row_tiles_per_image: int, shard_row_tiles: int, what: str) -> None:
    # The buffer distributes shards over the padded shape per image; it coincides with the 2-D view's
    # block grid (which the kernel indexes) only if no shard straddles an image boundary.
    if images > 1 and row_tiles_per_image % shard_row_tiles != 0:
        raise RuntimeError(
            f"g6_reshard: {what} shard height ({shard_row_tiles} tiles) does not divide the per-image height "
            f"({row_tiles_per_image} tiles) of a {images}-image tensor; its shards do not follow the 2-D view"
        )


def _plan(input_tensor, memory_config) -> ReshardPlan:
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshard: input_tensor must be on a device")
    _check_block_sharded_l1(input_tensor.memory_config(), "input_tensor")
    _check_block_sharded_l1(memory_config, "memory_config")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_reshard: input_tensor must be rank 4, got rank {len(shape)}")
    height = shape[0] * shape[1] * shape[2]
    if height % TILE_DIM != 0 or shape[-1] % TILE_DIM != 0:
        raise ValueError(f"g6_reshard: the 2-D view ({height}, {shape[-1]}) must have H and W multiples of {TILE_DIM}")

    out_spec = memory_config.shard_spec
    if int(out_spec.shape[0]) % TILE_DIM != 0 or int(out_spec.shape[1]) % TILE_DIM != 0:
        raise ValueError(
            f"g6_reshard: memory_config shard shape {list(out_spec.shape)} must be multiples of {TILE_DIM}"
        )
    if not _is_one_rectangle(out_spec.grid):
        raise ValueError(f"g6_reshard: memory_config core grid must be one rectangle, got {out_spec.grid}")

    in_spec = input_tensor.memory_config().shard_spec
    padded = input_tensor.padded_shape
    tensor_row_tiles, tensor_col_tiles, images = tile_grid(padded)
    out_placement = placement_from_shard_spec(out_spec)
    plan = ReshardPlan(
        tensor_row_tiles=tensor_row_tiles,
        tensor_col_tiles=tensor_col_tiles,
        tile_stride_bytes=input_tensor.buffer_aligned_page_size(),
        in_placement=placement_from_shard_spec(in_spec),
        out_placement=out_placement,
        num_issuers=NUM_ISSUERS,
    )

    x_span, y_span = out_placement.used_extent(*plan.out_blocks)
    if out_placement.grid_width < x_span or out_placement.grid_height < y_span:
        raise ValueError(
            f"g6_reshard: memory_config grid {out_placement.grid_width}x{out_placement.grid_height} cores "
            f"cannot hold {plan.out_blocks[0]}x{plan.out_blocks[1]} blocks (rows x cols) in its orientation"
        )

    device_grid = input_tensor.device().compute_with_storage_grid_size()
    if out_placement.x1 >= device_grid.x or out_placement.y1 >= device_grid.y:
        raise ValueError(
            f"g6_reshard: memory_config grid ends at ({out_placement.x1}, {out_placement.y1}), outside the "
            f"{device_grid.x}x{device_grid.y} worker grid"
        )

    out_shard_bytes = out_placement.shard_row_tiles * out_placement.shard_col_tiles * plan.tile_stride_bytes
    if out_shard_bytes > L1_BUDGET_BYTES:
        raise RuntimeError(
            f"g6_reshard: out shard of {out_shard_bytes} B exceeds the {L1_BUDGET_BYTES} B of L1 left to the op"
        )

    l1_alignment = ttnn.get_l1_alignment()
    if plan.tile_stride_bytes % l1_alignment != 0:
        raise RuntimeError(
            f"g6_reshard: tile stride {plan.tile_stride_bytes} B is not a multiple of the L1 alignment {l1_alignment} B"
        )

    row_tiles_per_image = int(padded[-2]) // TILE_DIM
    _check_2d_view_matches_buffer(images, row_tiles_per_image, plan.in_placement.shard_row_tiles, "input_tensor")
    _check_2d_view_matches_buffer(images, row_tiles_per_image, out_placement.shard_row_tiles, "memory_config")
    return plan


def g6_reshard_plan(input_tensor, memory_config) -> dict:
    """The plan a call selects (BB-08) and the role of every core of out's grid (BB-21)."""
    return _plan(input_tensor, memory_config).report()


# ---------------------------------------------------------------------------
# Program cache (BB-07)
# ---------------------------------------------------------------------------

# Cache key -> ProgramDescriptor. A hit only rewrites the two buffer addresses.
_DESCRIPTOR_CACHE = {}


def _device_key(device):
    try:
        return device.id()
    except (AttributeError, TypeError):
        return id(device)


def _cache_key(input_tensor, plan: ReshardPlan):
    return (
        _device_key(input_tensor.device()),
        tuple(int(d) for d in input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        plan,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    plan = _plan(input_tensor, memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config,
    )
    if output_tensor.buffer_aligned_page_size() != plan.tile_stride_bytes:
        raise RuntimeError(
            f"g6_reshard: out page stride {output_tensor.buffer_aligned_page_size()} B differs from the input's "
            f"{plan.tile_stride_bytes} B"
        )

    key = _cache_key(input_tensor, plan)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        patch_addresses(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
