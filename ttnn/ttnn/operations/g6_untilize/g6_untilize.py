# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize: uint32 TILE -> ROW_MAJOR, every logical element bit-identical, one device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
INVALID lives in eval/golden_tests/g6_untilize/feature_spec.py, not here.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_untilize_program_descriptor as pd

OP_NAME = "g6_untilize"


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs = (scenario,), scenario = {"input_shape", "dtype", "layout"}
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
        raise ValueError(f"{OP_NAME}: use_multicore must be True, got {use_multicore!r}")
    if sub_core_grids is not None:
        raise ValueError(f"{OP_NAME}: sub_core_grids must be None, got {sub_core_grids!r}")
    if not isinstance(input_tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError(f"{OP_NAME}: input_tensor must be a device tensor")
    shape = tuple(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"{OP_NAME}: input_tensor must have rank 4, got shape {shape}")
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"{OP_NAME}: input_tensor must be DRAM interleaved, got {mc}")

    scenario = (_scenario(input_tensor),)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(scenario, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Program cache (BB-07): one plan + descriptor parts per program-relevant key.
# A hit rewrites only the two buffer-address common runtime args.
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _cache_key(input_tensor, grid):
    mc = input_tensor.memory_config()
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        mc.memory_layout,
        mc.buffer_type,
        grid.x,
        grid.y,
        pd.knob_values(),
    )


class _CachedProgram:
    __slots__ = ("plan", "kernel", "cbs", "program_hash")

    def __init__(self, plan, kernel, cbs, program_hash):
        self.plan = plan
        self.kernel = kernel
        self.cbs = cbs
        self.program_hash = program_hash


def _plan_for(input_tensor, grid):
    return pd.make_plan(input_tensor.shape, input_tensor.element_size(), grid.x, grid.y)


def _cached_program(input_tensor, output_tensor):
    grid = input_tensor.device().compute_with_storage_grid_size()
    key = _cache_key(input_tensor, grid)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        plan = _plan_for(input_tensor, grid)
        kernel, cb = pd.create_program_parts(input_tensor, output_tensor, plan)
        descriptor = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        entry = _CachedProgram(plan, kernel, [cb], ttnn.compute_program_descriptor_hash(descriptor))
        _PROGRAM_CACHE[key] = entry
    return entry


def describe_plan(input_tensor) -> dict:
    """The plan a call with this input selects (BB-08 / BB-21): the same cached plan the entry point uses."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    entry = _PROGRAM_CACHE.get(_cache_key(input_tensor, grid))
    plan = entry.plan if entry is not None else _plan_for(input_tensor, grid)
    q_min, q_max = plan.blocks_per_core
    return {
        "regime": plan.regime,
        "core_ranges": [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in plan.core_ranges],
        "num_cores": plan.num_cores,
        "block_rows": plan.block_rows,
        "block_tile_cols": plan.block_tile_cols,
        "scratch_slots": plan.scratch_slots,
        "blocks_per_core": {"min": q_min, "max": q_max},
        "staging_bytes_per_core": plan.staging_bytes,
        "roles": {"row_gather": [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in plan.core_ranges]},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_untilize(input_tensor: ttnn.Tensor, *, use_multicore: bool = True, sub_core_grids=None) -> ttnn.Tensor:
    validate(input_tensor, use_multicore=use_multicore, sub_core_grids=sub_core_grids)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    entry = _cached_program(input_tensor, output_tensor)
    entry.kernel.common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    descriptor = ttnn.ProgramDescriptor(kernels=[entry.kernel], semaphores=[], cbs=entry.cbs)
    descriptor.custom_program_hash = entry.program_hash
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
