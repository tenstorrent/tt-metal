# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_untilize: uint32 TILE, DRAM-interleaved tensor -> new ROW_MAJOR tensor, bit-identical.

One generic_op dispatch of this op's own reader/writer program; byte moves only, no compute kernel.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_untilize_program_descriptor import describe_plan, get_program_descriptor


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


def validate(input_tensor, *, use_multicore=True, sub_core_grids=None):
    if use_multicore is not True:
        raise ValueError(f"g6_untilize: use_multicore must be True, got {use_multicore!r}")
    if sub_core_grids is not None:
        raise ValueError(f"g6_untilize: sub_core_grids must be None, got {sub_core_grids!r}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_untilize: input_tensor must be a device tensor")
    if len(input_tensor.shape) != 4:
        raise ValueError(f"g6_untilize: input_tensor must have rank 4, got rank {len(input_tensor.shape)}")
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(
            f"g6_untilize: input_tensor must be DRAM interleaved, got {mc.memory_layout} in {mc.buffer_type}"
        )
    # The kernels walk the 32x32 tile geometry of their CB; another input tile would mis-place every row,
    # and the descriptor cache key does not carry the tile shape.
    if input_tensor.layout == ttnn.TILE_LAYOUT and tuple(input_tensor.tile.tile_shape) != (32, 32):
        raise ValueError(f"g6_untilize: input_tensor must use 32x32 tiles, got {tuple(input_tensor.tile.tile_shape)}")

    scenario = {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_untilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_untilize: unsupported combination (refinement candidate): {exc}")


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
    descriptor = get_program_descriptor(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


__all__ = [
    "g6_untilize",
    "describe_plan",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]
