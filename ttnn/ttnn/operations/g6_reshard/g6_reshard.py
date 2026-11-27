# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move every tile of a BLOCK_SHARDED L1 tensor, as stored, into a new BLOCK_SHARDED L1 tensor on
the placement memory_config names. One device program (PULL regime, see op_design.md)."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_reshard_program_descriptor import (
    ISSUERS_PER_CORE,
    L1_FREE_BYTES,
    TILE_DIM,
    compute_geometry,
    create_program_descriptor,
    issuer_roles,
    set_buffer_addresses,
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


def validate(input_tensor, memory_config=None):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_reshard: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_reshard: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Input validation (beyond the support contract)
# ---------------------------------------------------------------------------


def _is_block_sharded_l1(memory_config) -> bool:
    return (
        memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        and memory_config.buffer_type == ttnn.BufferType.L1
        and memory_config.shard_spec is not None
    )


def _check_inputs(input_tensor, memory_config):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_reshard: input_tensor must be on a device")
    if not _is_block_sharded_l1(input_tensor.memory_config()):
        raise ValueError("g6_reshard: input_tensor must be BLOCK_SHARDED in L1 with a shard spec")
    if not isinstance(memory_config, ttnn.MemoryConfig) or not _is_block_sharded_l1(memory_config):
        raise ValueError("g6_reshard: memory_config must be BLOCK_SHARDED in L1 with a shard spec")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_reshard: input_tensor must be rank 4, got rank {len(shape)}")
    height = shape[0] * shape[1] * shape[2]
    if height % TILE_DIM or shape[3] % TILE_DIM:
        raise ValueError(f"g6_reshard: 2-D view ({height}, {shape[3]}) must have H and W multiples of {TILE_DIM}")

    spec = memory_config.shard_spec
    sh, sw = spec.shape[0], spec.shape[1]
    if sh <= 0 or sw <= 0 or sh % TILE_DIM or sw % TILE_DIM:
        raise ValueError(f"g6_reshard: memory_config shard shape ({sh}, {sw}) must be positive multiples of {TILE_DIM}")
    ranges = spec.grid.ranges()
    if len(ranges) != 1:
        raise ValueError("g6_reshard: memory_config core grid must be one rectangle")
    rng = ranges[0]
    grid = input_tensor.device().compute_with_storage_grid_size()
    if rng.end.x >= grid.x or rng.end.y >= grid.y:
        raise ValueError(
            f"g6_reshard: memory_config grid ({rng.start.x},{rng.start.y})-({rng.end.x},{rng.end.y}) "
            f"outside the {grid.x}x{grid.y} worker grid"
        )

    geom = compute_geometry(input_tensor, memory_config)
    grid_cols = rng.end.x - rng.start.x + 1
    grid_rows = rng.end.y - rng.start.y + 1
    need_cols, need_rows = (
        (geom.num_block_cols, geom.num_block_rows) if geom.out.row_major else (geom.num_block_rows, geom.num_block_cols)
    )
    if grid_cols < need_cols or grid_rows < need_rows:
        raise ValueError(
            f"g6_reshard: memory_config grid {grid_cols}x{grid_rows} (cols x rows) cannot hold "
            f"{geom.num_block_rows}x{geom.num_block_cols} blocks in its orientation"
        )
    if geom.out_shard_bytes > L1_FREE_BYTES:
        raise RuntimeError(
            f"g6_reshard: out shard of {geom.out_shard_bytes} B exceeds the {L1_FREE_BYTES} B of L1 left to the op"
        )
    return geom


# ---------------------------------------------------------------------------
# Host plan cache (BB-07) and plan report (BB-08)
# ---------------------------------------------------------------------------

_PLAN_CACHE: dict = {}
_LAST_PLAN: dict | None = None


def _plan_key(input_tensor, geom, device) -> tuple:
    return (
        tuple(input_tensor.padded_shape),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        geom.in0.key(),
        geom.out.key(),
        device.id(),
        ISSUERS_PER_CORE,
    )


def last_plan() -> dict | None:
    """Plan the last call selected: regime and the issuer group's core rectangle."""
    return _LAST_PLAN


def dump_roles(input_tensor, memory_config) -> list[dict]:
    """Per core of the kernel range: its out block and each issuer's row range and piece count (BB-21)."""
    return issuer_roles(compute_geometry(input_tensor, memory_config))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    global _LAST_PLAN

    geom = _check_inputs(input_tensor, memory_config)
    device = input_tensor.device()

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        device,
        memory_config,
    )

    key = _plan_key(input_tensor, geom, device)
    program_descriptor = _PLAN_CACHE.get(key)
    if program_descriptor is None:
        program_descriptor = create_program_descriptor(input_tensor, output_tensor, geom)
        program_descriptor.custom_program_hash = hash(("g6_reshard",) + key) & 0xFFFFFFFFFFFFFFFF
        _PLAN_CACHE[key] = program_descriptor
    else:
        set_buffer_addresses(program_descriptor, input_tensor.buffer_address(), output_tensor.buffer_address())

    _LAST_PLAN = {
        "regime": "PULL",
        "issuers_per_core": ISSUERS_PER_CORE,
        "issuer_rectangle": geom.holder_rectangle(),
        "out_blocks": (geom.num_block_rows, geom.num_block_cols),
    }

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
