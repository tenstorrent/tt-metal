# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros: a new DRAM-interleaved TILE tensor of ``shape``/``dtype`` filled with +0.0,
made by ONE device program (single writer kernel, bank-affine coalesced zero runs)."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_zeros_program_descriptor import create_program_descriptor, make_plan, num_dram_banks


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs[0] is the scenario dict {"shape", "dtype", "layout"}
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — bfloat16 / bfloat8_b are knob-turns: the dtype only enters via the page size (aps)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(shape, *, dtype, layout, device, memory_config):
    # (1) structural checks
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError(f"g6_zeros: shape must be 4 ints, got {shape!r}")
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in shape):
        raise ValueError(f"g6_zeros: shape must be ints, got {shape!r}")
    if any(d < 1 for d in shape):
        raise ValueError(f"g6_zeros: every dim must be >= 1, got {shape!r}")
    if shape[2] % 32 != 0 or shape[3] % 32 != 0:
        raise ValueError(f"g6_zeros: d_2 and d_3 must be multiples of 32, got {shape!r}")
    if layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_zeros: layout must be TILE_LAYOUT, got {layout!r}")
    if (
        memory_config is None
        or memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_zeros: memory_config must be DRAM interleaved, got {memory_config!r}")
    if device is None or not hasattr(device, "compute_with_storage_grid_size"):
        raise ValueError(f"g6_zeros: device must be an open ttnn device / mesh device, got {device!r}")

    # (2) axes from taggers
    scenario = {"shape": tuple(shape), "dtype": dtype, "layout": layout}
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    # (3) SUPPORTED per axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_zeros: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # (4) EXCLUSIONS
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_zeros: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Host-side descriptor cache (BB-07): build once per plan key, patch the address per call.
# ---------------------------------------------------------------------------

_DESCRIPTOR_CACHE = {}


def _plan_inputs(shape, dtype, device):
    """Host derivation of the plan inputs: N (tile pages), aps (tile bytes), grid, B.

    The single source the plan report and the dispatch path share; on a descriptor-cache miss the
    dispatch path cross-checks N and aps against the allocated buffer (the allocator stays the
    source of truth)."""
    d0, d1, d2, d3 = shape
    num_pages = d0 * d1 * (d2 // ttnn.TILE_SIZE) * (d3 // ttnn.TILE_SIZE)
    grid = device.compute_with_storage_grid_size()
    return num_pages, ttnn.tile_size(dtype), grid.x, grid.y, num_dram_banks(device)


def g6_zeros_plan(shape, dtype, device):
    """Report the selected plan (BB-08): regime R1, cores_per_bank, cores used, burst_pages, B.

    Pure host computation: allocates nothing on the device."""
    num_pages, aps, _, _, _ = _plan_inputs(shape, dtype, device)
    plan = make_plan(num_pages, aps, device)
    return {
        "regime": "R1_bank_affine_coalesced_runs",
        "num_pages": plan.num_pages,
        "num_banks": plan.num_banks,
        "cores_per_bank": plan.cores_per_bank,
        "cores_used": plan.num_cores,
        "core_ranges": str(plan.core_range_set()),
        "burst_pages": plan.burst_pages,
        "zero_source_bytes": plan.zero_source_bytes,
    }


def g6_zeros(shape, *, dtype, layout, device, memory_config):
    validate(shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)

    key = (str(dtype),) + _plan_inputs(shape, dtype, device)
    desc = _DESCRIPTOR_CACHE.get(key)
    if desc is None:
        num_pages, aps = key[1], key[2]
        assert num_pages == output.buffer_num_pages(), (num_pages, output.buffer_num_pages())
        assert aps == output.buffer_aligned_page_size(), (aps, output.buffer_aligned_page_size())
        plan = make_plan(num_pages, aps, device)
        desc = create_program_descriptor(output, plan)
        _DESCRIPTOR_CACHE[key] = desc
    else:
        # Only the buffer address changes between calls of one key (common RT arg 0).
        writer = desc.kernels[0]
        writer.common_runtime_args = [output.buffer_address()]
        desc.kernels = [writer]

    return ttnn.generic_op([output], desc)
