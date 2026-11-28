# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_copy(input_tensor, output_tensor): copy the source's values into the caller's destination
(same shape, dtype, layout, DRAM interleaved) in one device program and return the destination."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_copy_program_descriptor import get_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    if w % 32:
        return "w_non_aligned"
    if h % 32:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    # Every dtype and layout is the same page copy: page_bytes = buffer_aligned_page_size(),
    # and the CB format is the tensor's own, never interpreted (no compute path).
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.int32, ttnn.uint32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    {"dtype": ttnn.float32, "layout": ttnn.ROW_MAJOR_LAYOUT},
    {"dtype": ttnn.int32, "layout": ttnn.TILE_LAYOUT},
    {"dtype": ttnn.uint32, "layout": ttnn.TILE_LAYOUT},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_dram_interleaved(memory_config):
    return (
        memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and memory_config.buffer_type == ttnn.BufferType.DRAM
    )


def _check_request(input_tensor, output_tensor):
    for name, t in (("input_tensor", input_tensor), ("output_tensor", output_tensor)):
        if not ttnn.is_tensor_storage_on_device(t):
            raise ValueError(f"g6_copy: {name} must be on a device")
    rank = len(input_tensor.shape)
    if not 1 <= rank <= 4:
        raise ValueError(f"g6_copy: rank {rank} is outside 1 to 4")
    if tuple(input_tensor.shape) != tuple(output_tensor.shape):
        raise ValueError(
            f"g6_copy: shapes differ: input {tuple(input_tensor.shape)}, output {tuple(output_tensor.shape)}"
        )
    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(f"g6_copy: dtypes differ: input {input_tensor.dtype}, output {output_tensor.dtype}")
    if input_tensor.layout != output_tensor.layout:
        raise ValueError(f"g6_copy: layouts differ: input {input_tensor.layout}, output {output_tensor.layout}")
    input_memory, output_memory = input_tensor.memory_config(), output_tensor.memory_config()
    if input_memory != output_memory:
        raise ValueError(f"g6_copy: memory configs differ: input {input_memory}, output {output_memory}")
    if not _is_dram_interleaved(input_memory):
        raise ValueError(f"g6_copy: tensors must be DRAM interleaved, got {input_memory}")


def validate(input_tensor, output_tensor):
    _check_request(input_tensor, output_tensor)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_copy: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_copy: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_copy(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(input_tensor, output_tensor)
    descriptor = get_program_descriptor(input_tensor, output_tensor)
    ttnn.generic_op([input_tensor, output_tensor], descriptor)
    # The caller's object, not generic_op's wrapper: the model reads its own tensor.
    return output_tensor
