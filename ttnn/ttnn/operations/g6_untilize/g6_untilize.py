# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize: uint32 TILE -> ROW_MAJOR in new DRAM-interleaved storage, every element bit-identical.

One ttnn.generic_op dispatch per call. The kernel moves bytes over the NoC only (no engine), so every 32-bit
pattern survives.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_untilize_program_descriptor import REGIME, create_program_descriptor, plan


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# inputs is (scenario,) with scenario = {"input_shape", "dtype", "layout"} (feature_spec.py).


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
    "dtype": [ttnn.uint32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _scenario(input_tensor):
    return {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }


def validate(input_tensor, *, use_multicore=True, sub_core_grids=None):
    if use_multicore is not True:
        raise ValueError(f"g6_untilize: use_multicore must be True, got {use_multicore!r}")
    if sub_core_grids is not None:
        raise ValueError(f"g6_untilize: sub_core_grids must be None, got {sub_core_grids!r}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_untilize: input_tensor must be a device tensor")

    inputs = (_scenario(input_tensor),)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_untilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_untilize: unsupported combination (refinement candidate): {exc}")

    if len(input_tensor.shape) != 4:
        raise ValueError(f"g6_untilize: input rank must be 4, got rank {len(input_tensor.shape)}")

    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"g6_untilize: input must be DRAM interleaved, got {mc.memory_layout} in {mc.buffer_type}")


# ---------------------------------------------------------------------------
# Plan, descriptor cache
# ---------------------------------------------------------------------------

# key -> (ProgramDescriptor, KernelDescriptor). Every CT arg derives from the key; a hit patches only the two buffer
# addresses, which generic_op's program hash excludes (BB-07).
_DESCRIPTOR_CACHE = {}


def _grid(device):
    return device.compute_with_storage_grid_size()


def _mesh_shape(device):
    shape = getattr(device, "shape", None)
    return tuple(shape) if shape is not None else None


def _cache_key(input_tensor, device):
    grid = _grid(device)
    mc = input_tensor.memory_config()
    return (
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        mc.memory_layout,
        mc.buffer_type,
        (int(grid.x), int(grid.y)),
        _mesh_shape(device),
    )


def _plan_for(input_tensor):
    return plan(tuple(input_tensor.shape), input_tensor.element_size(), _grid(input_tensor.device()))


def describe_plan(input_tensor) -> dict:
    """The plan a call on input_tensor selects (BB-08) and each core's role (BB-21)."""
    p = _plan_for(input_tensor)
    roles = {}
    for i in range(p.num_cores):
        x, y = p.x0 + i % p.cx, p.y0 + i // p.cx
        start, n = p.rows_of_core(i)
        roles[(x, y)] = (REGIME, start, n)
    return {
        "regime": REGIME,
        "rectangle": {"x0": p.x0, "y0": p.y0, "cx": p.cx, "cy": p.cy},
        "knobs": {
            "block_rows": p.block_rows,
            "block_tile_cols": p.block_tile_cols,
            "staging_depth": p.staging_depth,
            "rotate_tile_order": p.rotate_tile_order,
        },
        "staging_bytes_per_core": p.staging_bytes,
        "cores": roles,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_untilize(
    input_tensor: ttnn.Tensor,
    *,
    use_multicore: bool = True,
    sub_core_grids: ttnn.CoreRangeSet = None,
) -> ttnn.Tensor:
    validate(input_tensor, use_multicore=use_multicore, sub_core_grids=sub_core_grids)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, device)
    cached = _DESCRIPTOR_CACHE.get(key)
    if cached is None:
        descriptor, kernel = create_program_descriptor(_plan_for(input_tensor), input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[key] = (descriptor, kernel)
    else:
        descriptor, kernel = cached
        # ProgramDescriptor.kernels hands out copies: patch the held kernel, then store it back.
        kernel.common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
        descriptor.kernels = [kernel]

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
