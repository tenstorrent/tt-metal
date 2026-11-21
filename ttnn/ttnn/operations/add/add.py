# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""add: `out = a + b`, elementwise (specification: eval/specs/add/spec.py).

Registry model: INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS and validate(), then the
public entry point, which validates and dispatches exactly one `ttnn.generic_op` (Metal 2.0 ProgramSpec,
add_program_artifacts.py).
"""

from __future__ import annotations

import torch

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.add.spec import AXES

from .add_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (= the specification's PHASE0, axis for axis).
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
    "output_dtype":     [None],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R2): defined here, and only here."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(input_tensor_a, input_tensor_b, *, dtype=None, compute_kernel_config=None):
    """Refuse anything outside SUPPORTED / inside EXCLUSIONS (NotImplementedError subclasses, R6), and shapes
    that do not broadcast (ValueError, rejection X1). Reads neither math_fidelity nor math_approx_mode (R3)."""
    config = _resolve_config(compute_kernel_config)
    shape_a = tuple(input_tensor_a.shape)
    b_is_number = _is_number(input_tensor_b)
    # A number operand has the shape (): its broadcast axis is "number", refused before any tensor attribute
    # of b is touched.
    shape_b = () if b_is_number else tuple(input_tensor_b.shape)
    if not b_is_number:
        try:
            torch.broadcast_shapes(shape_a, shape_b)
        except RuntimeError as e:
            raise ValueError(f"add: shapes {list(shape_a)} and {list(shape_b)} do not broadcast") from e

    axes = {
        "dtype":            input_tensor_a.dtype,
        "layout":           input_tensor_a.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "output_dtype":     dtype,
        **memory_axes(input_tensor_a),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape_a, shape_b), axes)

    # R1: float32 with a 16-bit DEST is legal but lossy, and refused.
    if axes["dtype"] == ttnn.float32 and not axes["fp32_dest_acc_en"]:
        raise UnsupportedAxisValue("add: float32 operands need fp32_dest_acc_en=True (rule R1)")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"add: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # Both operands are of the cell's dtype, layout and placement: b is checked against the same lists.
    if not b_is_number:
        b_axes = {"dtype": input_tensor_b.dtype, "layout": input_tensor_b.layout, **memory_axes(input_tensor_b)}
        for axis, value in b_axes.items():
            if value not in SUPPORTED[axis]:
                raise UnsupportedAxisValue(f"add: input_tensor_b {axis}={value!r} not in SUPPORTED {SUPPORTED[axis]}")
        if b_axes["dtype"] != axes["dtype"] or b_axes["layout"] != axes["layout"]:
            raise UnsupportedAxisValue(
                f"add: operands of different dtype/layout ({axes['dtype']}/{axes['layout']} and "
                f"{b_axes['dtype']}/{b_axes['layout']}) are not supported"
            )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"add: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def add(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor | float,
    *,
    dtype: ttnn.DataType = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(input_tensor_a, input_tensor_b, dtype=dtype, compute_kernel_config=compute_kernel_config)
    config = _resolve_config(compute_kernel_config)
    # R9: the output lives where the operands live (a's spec carries its memory config), interleaved.
    output = ttnn.allocate_tensor_on_device(input_tensor_a.spec, input_tensor_a.device())
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor_a, input_tensor_b, output, config)
    return ttnn.generic_op([input_tensor_a, input_tensor_b, output], spec, run_args, tensor_indices)
