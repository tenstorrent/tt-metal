# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast — convert every value of one tensor to `dtype` in exactly one device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
INVALID lives in eval/golden_tests/g6_typecast/feature_spec.py, never here.

The output keeps the input's shape, layout and memory configuration. One public call is one
`ttnn.generic_op` dispatch of the program built by g6_typecast_program_descriptor.py.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_typecast_program_descriptor import get_program, last_plan  # noqa: F401  (last_plan re-exported, BB-08)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs is (scenario,), scenario a dict with the keys
#    "input_shape", "dtype", "output_dtype", "layout", "buffer".
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_output_dtype(inputs, axes):
    return inputs[0]["output_dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


def tag_alignment(inputs, axes):
    shape = tuple(inputs[0]["input_shape"])
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    if w % 32 != 0:
        return "w_non_aligned"
    if h % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(tuple(inputs[0]["input_shape"]))


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "output_dtype": tag_output_dtype,
    "layout": tag_layout,
    "buffer": tag_buffer,
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0: bfloat16 TILE -> float32 / bfloat8_b / int32, DRAM or L1, rank 3-4.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "output_dtype": [ttnn.float32, ttnn.bfloat8_b, ttnn.int32],
    "layout": [ttnn.TILE_LAYOUT],
    "buffer": ["dram", "l1"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — stays empty in every phase (spec rule: every TARGET cell is a production case).
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_name(tensor):
    return "l1" if tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def validate(input_tensor, dtype, *, sub_core_grids=None):
    # Argument contract (ValueError), before the support check.
    if not isinstance(dtype, ttnn.DataType):
        raise ValueError(f"g6_typecast: dtype must be a ttnn.DataType, got {dtype!r}")
    if sub_core_grids is not None:
        raise ValueError("g6_typecast: sub_core_grids must be None")
    rank = len(tuple(input_tensor.shape))
    if rank < 1 or rank > 4:
        raise ValueError(f"g6_typecast: rank must be 1..4, got {rank}")

    scenario = {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "output_dtype": dtype,
        "layout": input_tensor.layout,
        "buffer": _buffer_name(input_tensor),
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_typecast: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_typecast: unsupported combination (refinement candidate): {exc}")

    # Interleaved placement only (TARGET has no sharded value).
    if input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("g6_typecast: only interleaved memory layouts are supported")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_typecast(input_tensor: ttnn.Tensor, dtype: ttnn.DataType, *, sub_core_grids=None) -> ttnn.Tensor:
    validate(input_tensor, dtype, sub_core_grids=sub_core_grids)

    device = input_tensor.device()
    # Output: the input's shape, layout and memory configuration (positional args only).
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    program_descriptor = get_program(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
