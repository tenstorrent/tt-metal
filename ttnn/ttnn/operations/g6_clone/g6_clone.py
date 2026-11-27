# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_clone: out = x.clone() as one device program of NoC byte moves.

The output is a newly allocated DRAM-interleaved tensor with x's TensorSpec; every page of x is
copied into the same page id of out. A full page moves whole; a partial TILE page (partial last
tile row or column) moves only its logical face rows, so out's tile padding is left as allocated.
No compute kernel runs, so every bit pattern of every dtype survives.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_clone_program_descriptor import compute_plan, create_program_descriptor, patch_addresses, tile_geometry


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is (scenario,), scenario = {"input_shape", "dtype", "layout"}
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
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.int32, ttnn.uint32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
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
    if not 1 <= len(input_tensor.shape) <= 4:
        raise ValueError(f"g6_clone: input_tensor rank must be 1 to 4, got {len(input_tensor.shape)}")
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
# Plan report and program cache
# ---------------------------------------------------------------------------

# Everything in a cached descriptor is a function of this key; only the two buffer addresses
# vary per call and are patched in place. The tile geometry selects the regime and the
# logical-rows pieces: [1,1,1,62080] and [1,1,32,62080] share page count and page size.
_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    memory_config = input_tensor.memory_config()
    return (
        device.id(),
        grid.x,
        grid.y,
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        input_tensor.buffer_num_pages(),
        input_tensor.buffer_aligned_page_size(),
        tile_geometry(input_tensor),
    )


def _plan(input_tensor):
    # The one place the plan's inputs are read off the tensor: the entry point and the plan report
    # must agree on them (BB-08), and every input that changes the plan must also be in _cache_key.
    return compute_plan(
        input_tensor.device(),
        input_tensor.buffer_num_pages(),
        input_tensor.buffer_aligned_page_size(),
        tile_geometry(input_tensor),
    )


def g6_clone_plan(input_tensor: ttnn.Tensor) -> dict:
    """The plan a call on `input_tensor` runs: regime, grid, work split, block knobs, per-core roles."""
    validate(input_tensor)
    return _plan(input_tensor)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_clone(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(input_tensor)

    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    # Same spec, so page p of x is page p of out.
    assert output_tensor.buffer_num_pages() == input_tensor.buffer_num_pages()
    assert output_tensor.buffer_aligned_page_size() == input_tensor.buffer_aligned_page_size()

    key = _cache_key(input_tensor)
    descriptor = _DESCRIPTOR_CACHE.get(key)
    if descriptor is None:
        descriptor = create_program_descriptor(input_tensor, output_tensor, _plan(input_tensor))
        _DESCRIPTOR_CACHE[key] = descriptor
    else:
        patch_addresses(descriptor, input_tensor, output_tensor)

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
