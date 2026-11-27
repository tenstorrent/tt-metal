# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros_like: +0.0 into every page of a DRAM-interleaved TILE tensor shaped like the input.

Matches torch.zeros_like(x), and torch.ops.aten.zeros_like.out(x, out=optional_tensor) when
optional_tensor is given. The input's data is never read: only its metadata shapes the program.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_zeros_like_program_descriptor as _pd

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# `inputs` is (scenario,), scenario = {"input_shape", "dtype", "with_optional_tensor"}.


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat8_b],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_device_tensor(t):
    return isinstance(t, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(t)


def _is_dram_interleaved(memory_config):
    return (
        memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and memory_config.buffer_type == ttnn.BufferType.DRAM
    )


def _validate_structure(input_tensor, optional_tensor=None):
    """Checks that hold for every TARGET cell; no SUPPORTED gate."""
    if not _is_device_tensor(input_tensor):
        raise ValueError("g6_zeros_like: input_tensor must be a ttnn.Tensor on device")

    shape = tuple(input_tensor.shape)
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"g6_zeros_like: input_tensor layout must be TILE, got {input_tensor.layout}")
    if len(shape) != 4:
        raise ValueError(f"g6_zeros_like: input_tensor must be rank 4, got shape {shape}")
    if shape[-1] % 32 != 0 or shape[-2] % 32 != 0:
        raise ValueError(f"g6_zeros_like: last two dims of input_tensor must be multiples of 32, got {shape}")
    if input_tensor.logical_volume() == 0:
        raise ValueError(f"g6_zeros_like: input_tensor has zero volume, shape {shape}")

    if not _is_dram_interleaved(input_tensor.memory_config()):
        raise ValueError(f"g6_zeros_like: input_tensor must be DRAM interleaved, got {input_tensor.memory_config()}")

    if optional_tensor is not None:
        if not _is_device_tensor(optional_tensor):
            raise ValueError("g6_zeros_like: optional_tensor must be a ttnn.Tensor on device")
        for name, got, want in (
            ("shape", tuple(optional_tensor.shape), shape),
            ("dtype", optional_tensor.dtype, input_tensor.dtype),
            ("layout", optional_tensor.layout, input_tensor.layout),
            ("memory_config", optional_tensor.memory_config(), input_tensor.memory_config()),
        ):
            if got != want:
                raise ValueError(f"g6_zeros_like: optional_tensor {name} {got} != input_tensor {name} {want}")


def validate(input_tensor, *, optional_tensor=None):
    _validate_structure(input_tensor, optional_tensor)

    scenario = {
        "input_shape": tuple(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "with_optional_tensor": optional_tensor is not None,
    }
    inputs = (scenario,)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_zeros_like: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_zeros_like: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Plan report
# ---------------------------------------------------------------------------


def describe_plan(input_tensor):
    """One entry per participating core; cores not listed carry no kernel.

    A core writes pages start_page + k * page_stride for k < num_pages; `bank` is the DRAM bank of all of
    them in the bank_run_coalesced regime, None in the page split.
    """
    _validate_structure(input_tensor)
    plan = _pd.plan(input_tensor)
    return [
        {
            "core": (c.core.x, c.core.y),
            "role": "writer",
            "regime": plan.regime,
            "start_page": c.start_page,
            "num_pages": c.num_pages,
            "page_stride": c.page_stride,
            "bank": c.bank,
        }
        for c in plan.cores
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_zeros_like(input_tensor: ttnn.Tensor, *, optional_tensor: ttnn.Tensor = None) -> ttnn.Tensor:
    validate(input_tensor, optional_tensor=optional_tensor)

    if optional_tensor is not None:
        output_tensor = optional_tensor
    else:
        output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())

    descriptor = _pd.get_program_descriptor(output_tensor)
    # The output is the only io tensor: the input's buffer is never bound to the program.
    ttnn.generic_op([output_tensor], descriptor)
    return output_tensor
