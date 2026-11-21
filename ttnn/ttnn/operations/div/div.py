# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""div: `torch.div(a, b, rounding_mode=...)` on Quasar, one `ttnn.ProgramSpec` dispatch per call.

The quotient, and its floor or trunc, are computed in DEST in one fused pass per tile
(op_design.md, Blocking Model). The divisor is a tensor of a's shape or a Python number; a number
travels to the compute kernel as a runtime argument (its float32 bit pattern) and is filled into DEST.
"""

from __future__ import annotations

import math
import numbers

import torch

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.div.spec import AXES

from .div_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, imported, never redefined
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — equals the specification's PHASE0 after the first phase
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved"],
    "alignment":        ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":             [2, 3, 4],
    "broadcast":        ["none", "number"],
    "rounding_mode":    [None, "floor", "trunc"],
    "output_dtype":     [None, ttnn.bfloat16, ttnn.float32],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS: list[dict] = []

ROUNDING_MODES = (None, "floor", "trunc")


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R3): the one place the default lives."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_config(compute_kernel_config):
    """`compute_kernel_config`, with None resolved through the one default factory (rule R3)."""
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


def _is_number(value) -> bool:
    # R8: an int (bool included) is the number divisor exactly like the float of the same value; a numpy
    # scalar is a number too (numbers.Real covers int, float, bool and numpy's real scalars).
    return isinstance(value, numbers.Real)


def _number_value(value) -> float:
    """The number divisor as a Python float. An int past the float range saturates to +-inf (the float32
    transport saturates the same way past FLT_MAX) instead of raising."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def validate(input_tensor_a, input_tensor_b, *, rounding_mode=None, dtype=None, compute_kernel_config=None):
    """The runtime gate (rule R12). Reads neither `math_fidelity` nor `math_approx_mode` (rule R4)."""
    # X2: a malformed rounding mode.
    if rounding_mode not in ROUNDING_MODES:
        raise ValueError(f"div: rounding_mode must be None, 'floor' or 'trunc', got {rounding_mode!r}")

    number = _is_number(input_tensor_b)
    a_shape = tuple(input_tensor_a.shape)
    b_shape = () if number else tuple(input_tensor_b.shape)

    # X1: shapes that do not broadcast.
    if not number:
        try:
            torch.broadcast_shapes(a_shape, b_shape)
        except RuntimeError as err:
            raise ValueError(f"div: shapes {list(a_shape)} and {list(b_shape)} do not broadcast") from err

    fp32_dest = bool(_resolve_config(compute_kernel_config).fp32_dest_acc_en)

    # R1 / R2: hold in every phase, independently of SUPPORTED.
    if input_tensor_a.dtype == ttnn.float32 and not fp32_dest:
        raise UnsupportedAxisValue("div: float32 operands need fp32_dest_acc_en=True (rule R1)")
    if dtype == ttnn.float32 and not fp32_dest:
        raise UnsupportedAxisValue("div: dtype=float32 needs fp32_dest_acc_en=True (rule R2)")

    # Both tensor operands share one dtype and one layout (mixed operands are not in TARGET).
    if not number:
        if input_tensor_b.dtype != input_tensor_a.dtype:
            raise UnsupportedAxisValue(
                f"div: operands of different dtypes ({input_tensor_a.dtype}, {input_tensor_b.dtype})"
            )
        if input_tensor_b.layout != input_tensor_a.layout:
            raise UnsupportedAxisValue(
                f"div: operands of different layouts ({input_tensor_a.layout}, {input_tensor_b.layout})"
            )

    # R14: every tensor operand's placement is derived with memory_axes and gated, b's as well as a's.
    placements = [memory_axes(input_tensor_a)] + ([] if number else [memory_axes(input_tensor_b)])
    axes = {
        "dtype":            input_tensor_a.dtype,
        "layout":           input_tensor_a.layout,
        "fp32_dest_acc_en": fp32_dest,
        "rounding_mode":    rounding_mode,
        "output_dtype":     dtype,
        **placements[0],
    }
    inputs = (a_shape, b_shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for placed in placements:
        for axis in ("memory_config", "sharding"):
            if placed[axis] not in SUPPORTED[axis]:
                raise UnsupportedAxisValue(f"div: {axis}={placed[axis]!r} not in SUPPORTED {SUPPORTED[axis]}")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"div: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"div: unsupported combination (refinement candidate): {exc}")

    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def div(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor | float,
    *,
    rounding_mode: str = None,
    dtype: ttnn.DataType = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """`torch.div(a, b, rounding_mode=rounding_mode)`; b a tensor or a Python number."""
    validate(
        input_tensor_a,
        input_tensor_b,
        rounding_mode=rounding_mode,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )
    config = _resolve_config(compute_kernel_config)

    number = _is_number(input_tensor_b)
    a_shape = tuple(input_tensor_a.shape)
    out_shape = a_shape if number else tuple(torch.broadcast_shapes(a_shape, tuple(input_tensor_b.shape)))
    out_dtype = dtype if dtype is not None else input_tensor_a.dtype

    # R13: the output lives in a's buffer, interleaved.
    out_memory = ttnn.L1_MEMORY_CONFIG if memory_axes(input_tensor_a)["memory_config"] == "L1" else ttnn.DRAM_MEMORY_CONFIG
    device = input_tensor_a.device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(list(out_shape)), out_dtype, ttnn.TILE_LAYOUT, out_memory), device
    )

    tensors = [input_tensor_a] + ([] if number else [input_tensor_b]) + [output]
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor_a,
        None if number else input_tensor_b,
        output,
        divisor=_number_value(input_tensor_b) if number else None,
        rounding_mode=rounding_mode,
        compute_kernel_config=config,
    )
    return ttnn.generic_op(tensors, spec, run_args, tensor_indices)
