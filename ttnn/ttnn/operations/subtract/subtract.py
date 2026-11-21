# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""subtract: `out = a - b` elementwise, one Metal 2.0 `ttnn.ProgramSpec` dispatch on Quasar.

Specification: eval/specs/subtract/spec.py. Design: op_design.md in this directory.
Phase 0 builds regime R0 (`tile_stream_same_shape`): two bfloat16 TILE operands of one shape, both in DRAM
or both in L1, interleaved; the flat tile range is split over the grid and each cluster runs one `sub`
helper call over its contiguous range (subtract_program_artifacts.py).
"""

from __future__ import annotations

import torch

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.subtract.spec import AXES

from .subtract_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0 equals the specification's PHASE0, axis for axis
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved"],
    "alignment":        ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":             [2, 3, 4],
    "broadcast":        ["none"],
    "output_dtype":     [None, ttnn.bfloat16],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused for now.
# R1 and R2 of the specification: legal but lossy, refused natively; kept here so they stay refused
# when float32 / fp32_dest_acc_en=False enter SUPPORTED.
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},         # R1
    {"output_dtype": ttnn.float32, "fp32_dest_acc_en": False},  # R2
]


# ---------------------------------------------------------------------------
# Compute config: the one place the default lives (rule R3)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (the specification's CALL default_factory)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )


def _resolve_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _broadcast_shape(a_shape, b_shape):
    try:
        return list(torch.broadcast_shapes(tuple(a_shape), tuple(b_shape)))
    except RuntimeError as exc:
        # X1: the shapes differ in a dimension where neither has size 1.
        raise ValueError(f"subtract: shapes {tuple(a_shape)} and {tuple(b_shape)} do not broadcast") from exc


def _axes_of(input_tensor_a, input_tensor_b, dtype, config):
    a_shape, b_shape = tuple(input_tensor_a.shape), tuple(input_tensor_b.shape)
    _broadcast_shape(a_shape, b_shape)
    if input_tensor_a.dtype != input_tensor_b.dtype or input_tensor_a.layout != input_tensor_b.layout:
        # Both operands carry the cell's dtype and layout (spec CALL); mixed operands are not a cell.
        raise UnsupportedAxisValue(
            f"subtract: operands differ in dtype or layout ({input_tensor_a.dtype}/{input_tensor_a.layout} vs "
            f"{input_tensor_b.dtype}/{input_tensor_b.layout})"
        )
    placed_a, placed_b = memory_axes(input_tensor_a), memory_axes(input_tensor_b)  # R10
    if placed_a != placed_b:
        raise UnsupportedAxisValue(f"subtract: operands in different buffers ({placed_a} vs {placed_b})")
    axes = {
        "dtype": input_tensor_a.dtype,
        "layout": input_tensor_a.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "output_dtype": dtype,
        "memory_config": placed_a["memory_config"],
        "sharding": placed_a["sharding"],
    }
    inputs = (a_shape, b_shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(input_tensor_a, input_tensor_b, *, dtype=None, compute_kernel_config=None):
    """Raise for anything outside the current phase. Reads neither math_fidelity nor math_approx_mode (R4)."""
    config = _resolve_config(compute_kernel_config)
    axes = _axes_of(input_tensor_a, input_tensor_b, dtype, config)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"subtract: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"subtract: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point (signature = the specification's stub)
# ---------------------------------------------------------------------------


def subtract(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    dtype: ttnn.DataType = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(input_tensor_a, input_tensor_b, dtype=dtype, compute_kernel_config=compute_kernel_config)
    config = _resolve_config(compute_kernel_config)

    device = input_tensor_a.device()
    out_shape = _broadcast_shape(input_tensor_a.shape, input_tensor_b.shape)
    out_dtype = dtype if dtype is not None else input_tensor_a.dtype
    # R9: the output lives in the first operand's buffer, interleaved.
    out_spec = ttnn.TensorSpec(
        ttnn.Shape(out_shape), out_dtype, input_tensor_a.layout, input_tensor_a.memory_config()
    )
    output = ttnn.allocate_tensor_on_device(out_spec, device)

    spec, run_args, tensor_indices = create_program_artifacts(input_tensor_a, input_tensor_b, output, config)
    return ttnn.generic_op([input_tensor_a, input_tensor_b, output], spec, run_args, tensor_indices)
