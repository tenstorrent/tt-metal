# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_clone: bit-exact copy of a DRAM-interleaved device tensor into a new buffer.

out = x.clone(): same TensorSpec, new storage, every stored byte moved by NoC
reads (reader, NCRISC/NoC0) into one staging CB and NoC writes (writer,
BRISC/NoC1) out of it. No compute kernel: an unpack/DEST/pack path would change
float32 bits (NaN payloads, subnormals, low mantissa bits).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_clone_program_descriptor import (
    create_program_descriptor,
    g6_clone_plan,
    plan_key,
    set_buffer_addresses,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# inputs is (scenario,), scenario = {"input_shape", "dtype", "layout"}.


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


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor):
    if not isinstance(input_tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_clone: input_tensor must be a device tensor")
    rank = len(input_tensor.shape)
    if not 1 <= rank <= 4:
        raise ValueError(f"g6_clone: input_tensor rank must be 1 to 4, got {rank}")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(f"g6_clone: input_tensor must be DRAM interleaved, got {memory_config}")

    scenario = {
        "input_shape": tuple(int(d) for d in input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_clone: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_clone: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# plan_key -> (ProgramDescriptor, plan). Everything in the descriptor is a function
# of the key; only the two buffer addresses change between calls and are patched.
_PROGRAM_CACHE = {}


def g6_clone(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(input_tensor)

    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    if output_tensor.buffer_num_pages() != input_tensor.buffer_num_pages():
        raise RuntimeError("g6_clone: output page count differs from the input's")
    if output_tensor.buffer_aligned_page_size() != input_tensor.buffer_aligned_page_size():
        raise RuntimeError("g6_clone: output aligned page size differs from the input's")

    key = plan_key(input_tensor)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        plan = g6_clone_plan(input_tensor)
        cached = (create_program_descriptor(input_tensor, output_tensor, plan), plan)
        _PROGRAM_CACHE[key] = cached
    program_descriptor, _ = cached
    set_buffer_addresses(program_descriptor, input_tensor.buffer_address(), output_tensor.buffer_address())

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
