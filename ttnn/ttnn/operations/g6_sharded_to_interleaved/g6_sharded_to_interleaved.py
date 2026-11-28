# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved: a height- or width-sharded L1 TILE tensor copied, bit for bit, into a
newly allocated interleaved tensor in DRAM or L1, in one device program.

Regime `owner_writes`: every shard core writes its resident tiles straight to their output pages.
Design: op_design.md.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_sharded_to_interleaved_program_descriptor import (
    NUM_WRITE_LANES,
    TILE,
    build_plan,
    create_program_descriptor,
    patch_program_descriptor,
)

_IN_SCHEME_NAMES = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
}
_BUFFER_NAMES = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_ORIENTATION_NAMES = {
    ttnn.ShardOrientation.ROW_MAJOR: "row_major",
    ttnn.ShardOrientation.COL_MAJOR: "column_major",
}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,); see eval/golden_tests/g6_sharded_to_interleaved/feature_spec.py
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_in_scheme(inputs, axes):
    return inputs[0]["in_scheme"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "in_scheme": tag_in_scheme,
    "buffer": tag_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "in_scheme": ["width_sharded"],
    "buffer": ["dram"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _structural_checks(input_tensor, memory_config):
    """Malformed inputs raise ValueError, never a support refusal."""
    if not isinstance(input_tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_sharded_to_interleaved: input_tensor must be a ttnn.Tensor stored on a device")
    rank = len(input_tensor.shape)
    if rank not in (2, 4):
        raise ValueError(f"g6_sharded_to_interleaved: input rank {rank} not in (2, 4)")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_sharded_to_interleaved: input layout {input_tensor.layout} is not TILE_LAYOUT")
    in_mc = input_tensor.memory_config()
    if not in_mc.is_sharded() or in_mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"g6_sharded_to_interleaved: input must be sharded in L1, got {in_mc}")
    if in_mc.memory_layout not in _IN_SCHEME_NAMES:
        raise ValueError(f"g6_sharded_to_interleaved: input {in_mc.memory_layout} is not height or width sharded")
    shard_shape = in_mc.shard_spec.shape
    if shard_shape[0] % TILE or shard_shape[1] % TILE:
        raise ValueError(f"g6_sharded_to_interleaved: shard shape {tuple(shard_shape)} is not whole tiles")
    if (
        not isinstance(memory_config, ttnn.MemoryConfig)
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        or memory_config.buffer_type not in _BUFFER_NAMES
    ):
        raise ValueError(
            f"g6_sharded_to_interleaved: memory_config must be DRAM or L1 interleaved, got {memory_config}"
        )


def _scenario(input_tensor, memory_config):
    spec = input_tensor.memory_config().shard_spec
    return {
        "shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_scheme": _IN_SCHEME_NAMES[input_tensor.memory_config().memory_layout],
        "shard_shape": tuple(spec.shape),
        "grid": tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()),
        "orientation": _ORIENTATION_NAMES[spec.orientation],
        "buffer": _BUFFER_NAMES[memory_config.buffer_type],
    }


def validate(input_tensor, memory_config):
    """Gate the call; returns the scenario the axes were read from."""
    _structural_checks(input_tensor, memory_config)

    scenario = _scenario(input_tensor, memory_config)
    inputs = (scenario,)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_sharded_to_interleaved: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_sharded_to_interleaved: unsupported combination (refinement candidate): {exc}")

    return scenario


# ---------------------------------------------------------------------------
# Plan / descriptor memo (BB-07)
# ---------------------------------------------------------------------------

# key -> Plan, and key -> ProgramDescriptor. A hit patches the descriptor with the call's input
# buffer and output address instead of rebuilding it.
_PLANS = {}
_DESCRIPTORS = {}


def _device_key(device):
    if hasattr(device, "get_device_ids"):
        return ("mesh", tuple(device.shape), tuple(device.get_device_ids()))
    return ("device", device.id())


def _memo_key(input_tensor, scenario, num_write_lanes):
    # Orientation and range order change the per-core RT values without changing the C++ program
    # hash, so both must be in the key.
    return (
        _device_key(input_tensor.device()),
        tuple(input_tensor.padded_shape),
        scenario["shape"],
        scenario["dtype"],
        scenario["in_scheme"],
        scenario["shard_shape"],
        scenario["grid"],
        scenario["orientation"],
        scenario["buffer"],
        num_write_lanes,
    )


def _plan(key, input_tensor, memory_config, num_write_lanes):
    plan = _PLANS.get(key)
    if plan is None:
        plan = build_plan(input_tensor, memory_config, num_write_lanes)
        _PLANS[key] = plan
    return plan


def describe_plan(input_tensor, memory_config, num_write_lanes=NUM_WRITE_LANES):
    """Report the plan a call with these inputs selects (BB-08, BB-21)."""
    scenario = validate(input_tensor, memory_config)
    key = _memo_key(input_tensor, scenario, num_write_lanes)
    plan = _plan(key, input_tensor, memory_config, num_write_lanes)
    cores = []
    for i, (x, y) in enumerate(plan.cores):
        block = plan.blocks[i]
        cores.append(
            {
                "core": (x, y),
                "role": "idle" if block is None else "writer",
                "block": None if block is None else (block.first_row, block.first_col, block.rows, block.cols),
                "lanes": [
                    None if lb[i] is None else (lb[i].first_row, lb[i].first_col, lb[i].rows, lb[i].cols)
                    for lb in plan.lane_blocks
                ],
            }
        )
    return {
        "plan": "owner_writes",
        "num_write_lanes": plan.num_write_lanes,
        "tensor_tiles": (plan.tensor_tile_rows, plan.tensor_tile_cols),
        "shard_tiles": (plan.shard_tile_rows, plan.shard_tile_cols),
        "page_size": plan.page_size,
        "cores": cores,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _dispatch(input_tensor, memory_config, scenario, num_write_lanes):
    key = _memo_key(input_tensor, scenario, num_write_lanes)
    plan = _plan(key, input_tensor, memory_config, num_write_lanes)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        memory_config,
    )

    descriptor = _DESCRIPTORS.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(input_tensor, output_tensor, plan)
        _DESCRIPTORS[key] = descriptor
    else:
        patch_program_descriptor(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    scenario = validate(input_tensor, memory_config)
    return _dispatch(input_tensor, memory_config, scenario, NUM_WRITE_LANES)
