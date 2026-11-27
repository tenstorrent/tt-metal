# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros: a new DRAM-interleaved TILE tensor whose every element is +0.0.

One device program per call (regime `bank_run_write`): the output's pages are
split bank-major over the whole grid; each core makes its zero bytes once in
its own L1 and writes its <= 2 DRAM bank runs from them. Nothing is read,
nothing is made on the host.
"""

from __future__ import annotations

import operator

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_zeros_program_descriptor import (
    TILE_DIM,
    create_program_descriptor,
    device_geometry,
    make_plan,
    set_output_address,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — project the scenario dict {"shape", "dtype", "layout"}
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    """The dtype sets the page size (4096 / 2048 / 1088 B per tile); the bytes written are zero in all."""
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    """A page is one 32x32 tile."""
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
    for size in shape:
        if isinstance(size, bool):
            raise ValueError(f"g6_zeros: shape sizes must be ints, got {shape!r}")
        try:
            operator.index(size)
        except TypeError:
            raise ValueError(f"g6_zeros: shape sizes must be ints, got {shape!r}") from None
        if size < 1:
            raise ValueError(f"g6_zeros: shape sizes must be >= 1, got {shape!r}")
    if shape[2] % TILE_DIM or shape[3] % TILE_DIM:
        raise ValueError(f"g6_zeros: the last two sizes must be multiples of {TILE_DIM}, got {shape!r}")
    if layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_zeros: layout must be TILE_LAYOUT, got {layout}")
    if not isinstance(memory_config, ttnn.MemoryConfig) or not (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_zeros: memory_config must be DRAM interleaved, got {memory_config}")
    if not isinstance(device, ttnn.MeshDevice):
        raise ValueError(f"g6_zeros: device must be an open ttnn.MeshDevice, got {device!r}")


def validate(shape, *, dtype, layout, device, memory_config):
    _check_arguments(shape, layout, device, memory_config)

    scenario = {"shape": tuple(int(d) for d in shape), "dtype": dtype, "layout": layout}
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_zeros: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for excluded in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excluded.items()):
            raise ExcludedCell(f"g6_zeros: unsupported combination (refinement candidate): {excluded}")


# ---------------------------------------------------------------------------
# Host descriptor cache and plan report
# ---------------------------------------------------------------------------

# Key -> built ProgramDescriptor. Holds descriptors only, never a tensor: every call allocates its output.
_DESCRIPTOR_CACHE = {}


def _cache_key(shape, dtype, layout, memory_config, grid, num_dram_banks):
    return (
        tuple(int(d) for d in shape),
        dtype,
        layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        grid,
        num_dram_banks,
    )


def g6_zeros_plan(shape, *, dtype, device):
    """The plan a call with these arguments runs, without dispatching (BB-08, BB-21).

    Regime, grid, cores used, per-core role / position range / bank runs, zero source and packet size.
    """
    grid, num_dram_banks = device_geometry(device)
    return make_plan(tuple(int(d) for d in shape), dtype, grid, num_dram_banks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_zeros(shape, *, dtype, layout, device, memory_config):
    validate(shape, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, memory_config)

    grid, num_dram_banks = device_geometry(device)
    key = _cache_key(shape, dtype, layout, memory_config, grid, num_dram_banks)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        plan = make_plan(key[0], dtype, grid, num_dram_banks)
        descriptor = create_program_descriptor(output, plan)
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        set_output_address(descriptor, output)

    return ttnn.generic_op([output], descriptor)
