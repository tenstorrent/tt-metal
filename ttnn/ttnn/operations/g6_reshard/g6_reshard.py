# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard: move a BLOCK_SHARDED L1 tensor, tile for tile as stored, into a new BLOCK_SHARDED L1
tensor on the placement `memory_config` names. One device program (PULL regime, see op_design.md).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_reshard_program_descriptor as pd

# ---------------------------------------------------------------------------
# Registry declarations
# ---------------------------------------------------------------------------

# No shape-derived axis: the placements are arguments of every call, not axes.
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
# Argument checks and the plan
# ---------------------------------------------------------------------------

# L1 per core the model leaves to the op (out's shard is the op's only allocation).
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
    tile_h, tile_w = input_tensor.tile.tile_shape
    height = 1
    for d in shape[:-1]:
        height *= d
    if height % tile_h or shape[-1] % tile_w:
        raise ValueError(
            f"g6_reshard: 2-D view ({height}, {shape[-1]}) must be a multiple of the tile ({tile_h}, {tile_w})"
        )


def _check_memory_config(memory_config, device, tile_hw, tensor_rows_tiles, tensor_cols_tiles) -> None:
    if not isinstance(memory_config, ttnn.MemoryConfig) or not _is_block_sharded_l1(memory_config):
        raise ValueError("g6_reshard: memory_config must be BLOCK_SHARDED in L1 with a shard spec")
    spec = memory_config.shard_spec
    tile_h, tile_w = tile_hw
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    if shard_h <= 0 or shard_w <= 0 or shard_h % tile_h or shard_w % tile_w:
        raise ValueError(f"g6_reshard: memory_config shard ({shard_h}, {shard_w}) must be a multiple of the tile")
    box = spec.grid.bounding_box()
    grid_x = box.end.x - box.start.x + 1
    grid_y = box.end.y - box.start.y + 1
    if grid_x * grid_y != spec.grid.num_cores():
        raise ValueError("g6_reshard: memory_config core grid must be one rectangle")
    size = device.compute_with_storage_grid_size()
    if box.end.x >= size.x or box.end.y >= size.y:
        raise ValueError(
            f"g6_reshard: memory_config core grid ends at ({box.end.x}, {box.end.y}), outside the worker grid {size.x}x{size.y}"
        )
    block_rows = -(-tensor_rows_tiles // (shard_h // tile_h))
    block_cols = -(-tensor_cols_tiles // (shard_w // tile_w))
    need_x, need_y = (
        (block_rows, block_cols) if spec.orientation == ttnn.ShardOrientation.COL_MAJOR else (block_cols, block_rows)
    )
    if grid_x < need_x or grid_y < need_y:
        raise ValueError(
            f"g6_reshard: out has {block_rows} x {block_cols} blocks; memory_config grid {grid_x}x{grid_y} "
            f"({spec.orientation}) cannot hold them"
        )


def _plan(input_tensor, memory_config) -> pd.Plan:
    _check_input(input_tensor)
    device = input_tensor.device()
    tile_hw = tuple(input_tensor.tile.tile_shape)
    rows, cols = pd.tensor_tiles(list(input_tensor.padded_shape), tile_hw)
    _check_memory_config(memory_config, device, tile_hw, rows, cols)

    page_bytes = input_tensor.buffer_aligned_page_size()
    if page_bytes % ttnn.get_l1_alignment():
        raise RuntimeError(f"g6_reshard: page size {page_bytes} B is not a multiple of the L1 alignment")
    p_in = pd.placement_from_shard_spec(input_tensor.memory_config().shard_spec, rows, cols, tile_hw)
    p_out = pd.placement_from_shard_spec(memory_config.shard_spec, rows, cols, tile_hw)
    out_shard_bytes = p_out.shard_rows_tiles * p_out.shard_cols_tiles * page_bytes
    if out_shard_bytes > L1_FREE_BYTES:
        raise RuntimeError(
            f"g6_reshard: out shard {out_shard_bytes} B exceeds the {L1_FREE_BYTES} B of L1 left to the op"
        )
    return pd.make_plan(rows, cols, page_bytes, p_in, p_out)


# ---------------------------------------------------------------------------
# Descriptor cache: placements and knobs are in the key (and in the CT args); buffer addresses are
# not, they are the only values patched on a hit.
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _descriptor(plan, input_tensor, out, device):
    key = (device.id(), tuple(input_tensor.padded_shape), input_tensor.dtype, input_tensor.layout, plan)
    in_addr, out_addr = input_tensor.buffer_address(), out.buffer_address()
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = pd.create_program_descriptor(plan, device, in_addr, out_addr)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        pd.set_buffer_addresses(descriptor, in_addr, out_addr)
    return descriptor


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    validate(input_tensor, memory_config)
    plan = _plan(input_tensor, memory_config)
    device = input_tensor.device()

    # CRITICAL: allocate_tensor_on_device takes positional args.
    out = ttnn.allocate_tensor_on_device(
        input_tensor.shape, input_tensor.dtype, input_tensor.layout, device, memory_config
    )
    if out.buffer_aligned_page_size() != plan.page_bytes:
        raise RuntimeError(
            f"g6_reshard: out page {out.buffer_aligned_page_size()} B differs from in0 page {plan.page_bytes} B"
        )
    return ttnn.generic_op([input_tensor, out], _descriptor(plan, input_tensor, out, device))


# ---------------------------------------------------------------------------
# Plan / role dump (host arithmetic only, no dispatch)
# ---------------------------------------------------------------------------


def describe_plan(input_tensor, memory_config) -> dict:
    """The regime, the kernel core range, every target core's block and per-issuer rows, NoC and
    run count, and the in0 cores that only source data."""
    validate(input_tensor, memory_config)
    plan = _plan(input_tensor, memory_config)
    out_range = plan.p_out.trimmed_range()
    in_range = plan.p_in.trimmed_range()
    issuer_roles = [
        {"issuer": k, "processor": str(proc), "noc": str(noc)}
        for k, (proc, noc) in enumerate(pd._ISSUER_CONFIGS[: plan.issuers_per_core])
    ]

    targets = {}
    total_runs = 0
    for x in range(out_range.start.x, out_range.end.x + 1):
        for y in range(out_range.start.y, out_range.end.y + 1):
            r_out, c_out = plan.p_out.block_of(x, y)
            issuers = []
            for k in range(plan.issuers_per_core):
                runs = pd.issuer_runs(plan, r_out, c_out, k)
                total_runs += len(runs)
                issuers.append(
                    {
                        "rows": pd.issuer_rows(plan, r_out, k),
                        "noc": issuer_roles[k]["noc"],
                        "runs": len(runs),
                        "tiles": sum(t for _, t in runs),
                        "sources": sorted({plan.p_in.core_of(*src) for src, _ in runs}),
                    }
                )
            targets[(x, y)] = {"block": (r_out, c_out), "cols": pd.core_cols(plan, c_out), "issuers": issuers}

    source_only = [
        (x, y)
        for x in range(in_range.start.x, in_range.end.x + 1)
        for y in range(in_range.start.y, in_range.end.y + 1)
        if (x, y) not in targets
    ]
    return {
        "regime": "PULL",
        "issuers_per_core": plan.issuers_per_core,
        "issuer_roles": issuer_roles,
        "block_rows_tiles": plan.block_rows_tiles,
        "block_cols_tiles": plan.block_cols_tiles,
        "kernel_core_ranges": [((out_range.start.x, out_range.start.y), (out_range.end.x, out_range.end.y))],
        "target_cores": targets,
        "source_only_cores": {core: "source-only" for core in source_only},
        "total_runs": total_runs,
    }
