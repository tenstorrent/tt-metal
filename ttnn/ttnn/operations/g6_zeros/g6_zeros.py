# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros: a new DRAM-interleaved TILE tensor whose every element is +0.0.

One device program per call: every writer core zero-fills a 16 KiB L1 region, then streams a
contiguous same-bank run of output pages from it. Nothing is read from DRAM, and each page is
written once.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_zeros_program_descriptor import (
    create_program_descriptor,
    plan_bank_runs,
    set_output_address,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# `inputs` is (scenario,), scenario = {"shape": tuple, "dtype": ..., "layout": ...}.


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_arguments(shape, layout, device, memory_config):
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError(f"g6_zeros: shape must be a list or tuple of 4 ints, got {shape!r}")
    for d in shape:
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError(f"g6_zeros: shape sizes must be ints, got {shape!r}")
        if d < 1:
            raise ValueError(f"g6_zeros: shape sizes must be >= 1, got {shape!r}")
    if shape[2] % 32 != 0 or shape[3] % 32 != 0:
        raise ValueError(f"g6_zeros: the last two sizes must be multiples of 32, got {shape!r}")

    if layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_zeros: layout must be TILE_LAYOUT, got {layout!r}")

    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError(f"g6_zeros: memory_config must be a ttnn.MemoryConfig, got {memory_config!r}")
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_zeros: memory_config must be DRAM interleaved, got {memory_config!r}")

    if not isinstance(device, (ttnn.MeshDevice, ttnn.Device)):
        raise ValueError(f"g6_zeros: device must be a ttnn device or mesh device, got {device!r}")
    try:
        device.compute_with_storage_grid_size()
    except Exception as error:
        raise RuntimeError(f"g6_zeros: device is not open: {error}") from error


def validate(shape, *, dtype, layout, device, memory_config):
    _check_arguments(shape, layout, device, memory_config)

    scenario = {"shape": tuple(int(d) for d in shape), "dtype": dtype, "layout": layout}
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_zeros: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_zeros: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Host plan cache (BB-07, BB-08)
# ---------------------------------------------------------------------------
# One (plan, ProgramDescriptor) per key. A repeat call patches the output address and dispatches;
# it never rebuilds the descriptor. Every shape- or dtype-dependent value is in the key.

_PLAN_CACHE = {}


def _device_key(device):
    grid = device.compute_with_storage_grid_size()
    return (type(device).__name__, device.id(), grid.x, grid.y)


def _plan_key(shape, dtype, device):
    return (_device_key(device), tuple(int(d) for d in shape), dtype)


def _num_banks(device):
    return ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks


def _plan_for_output(output, device):
    return plan_bank_runs(
        output.buffer_num_pages(),
        _num_banks(device),
        output.buffer_aligned_page_size(),
        device.compute_with_storage_grid_size(),
    )


def describe_plan(shape, *, dtype, device) -> dict:
    """Report the plan a call with these arguments selects: regime, knobs, and every core's role."""
    key = _plan_key(shape, dtype, device)
    if key in _PLAN_CACHE:
        plan = _PLAN_CACHE[key][0]
    else:
        num_pages = int(shape[0]) * int(shape[1]) * -(-int(shape[2]) // 32) * -(-int(shape[3]) // 32)
        dram_alignment = ttnn.get_dram_alignment()
        aligned_page_size = -(-ttnn.tile_size(dtype) // dram_alignment) * dram_alignment
        plan = plan_bank_runs(num_pages, _num_banks(device), aligned_page_size, device.compute_with_storage_grid_size())

    roles = {}
    for x in range(plan.grid[0]):
        for y in range(plan.grid[1]):
            roles[(x, y)] = "idle"
    for core, core_blocks in zip(plan.cores, plan.blocks):
        roles[(core.x, core.y)] = [
            f"writer(bank={run.bank}, first_page_id={run.first_page_id}, core_slots={run.core_slots})"
            for run in core_blocks
        ]

    return {
        "regime": "bank_run",
        "num_pages": plan.num_pages,
        "N": plan.num_banks,
        "aligned_page_size": plan.aligned_page_size,
        "writers_per_bank": plan.writers_per_bank,
        "zero_source_bytes": plan.zero_source_bytes,
        "zero_source_regions": plan.zero_source_regions,
        "core_ranges": str(plan.core_ranges),
        "roles": roles,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_zeros(shape, *, dtype, layout, device, memory_config) -> ttnn.Tensor:
    validate(shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)

    key = _plan_key(shape, dtype, device)
    cached = _PLAN_CACHE.get(key)
    if cached is None:
        plan = _plan_for_output(output, device)
        cached = (plan, create_program_descriptor(output, plan))
        _PLAN_CACHE[key] = cached
    program_descriptor = cached[1]
    set_output_address(program_descriptor, output)

    return ttnn.generic_op([output], program_descriptor)
