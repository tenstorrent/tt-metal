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
from eval.specs.add.spec import AXES, canonical_memory

from .add_program_artifacts import create_program_artifacts, io_tensors, needs_broadcast

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (= the specification's PHASE0, axis for axis), widened by
#    Refinement 0:  "height" (both operands height-sharded in L1; R9, R10);
#    Refinement 20: "width" (both operands width-sharded in L1), memory_argument "input" (R11) and
#                   output_dtype bfloat16 (R8; the operands' own dtype);
#    Refinement 1:  broadcast "over_w", "over_h", "over_hw", "over_leading" (either operand may be the one that
#                   broadcasts, R4) and "number" (R7).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved", "height", "width"],
    "memory_argument":  [None, "input"],
    "alignment":        ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":             [2, 3, 4],
    "broadcast":        ["none", "over_w", "over_h", "over_hw", "over_leading", "number"],
    "output_dtype":     [None, ttnn.bfloat16],
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


def validate(input_tensor_a, input_tensor_b, *, dtype=None, memory_config=None, compute_kernel_config=None):
    """Refuse anything outside SUPPORTED / inside EXCLUSIONS (NotImplementedError subclasses, R6), and shapes
    that do not broadcast (ValueError, rejection X1). Reads neither math_fidelity nor math_approx_mode (R3).
    `memory_config` reads as the axis `memory_argument` through the specification's own canonicalization:
    None, "input" (a's own config) or "other", which no SUPPORTED list holds (R11)."""
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
        "memory_argument":  canonical_memory(memory_config, {"input_tensor_a": input_tensor_a}),
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

    # One operand must have the output's shape: FPU broadcast is srcB-only, so the host makes the broadcasting
    # operand srcB (R4); two operands that each broadcast somewhere would need it on both sides.
    if not b_is_number:
        out_shape = tuple(torch.broadcast_shapes(shape_a, shape_b))
        if needs_broadcast(shape_a, out_shape) and needs_broadcast(shape_b, out_shape):
            raise UnsupportedAxisValue(
                f"add: both operands broadcast ({list(shape_a)} and {list(shape_b)}); one must have the output's shape"
            )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"add: unsupported combination (refinement candidate): {exc}")


def _output_spec(a, b, dtype):
    """R9 / R8: the output lives where a lives, has the broadcast shape and the requested dtype (else a's).
    `memory_config=` can only be a's own config here (R11), so it changes nothing.

    Interleaved: a's memory config (a's buffer). Sharded: a's memory config (its shard shape, cores and
    orientation) when a has the output's shape up to leading 1s (the padded page sequences then match, so a's
    shard spec fits the output); when a is the operand that broadcasts, its shard spec cannot hold the output and
    the output takes the config of b, the operand of the output's shape (both operands of a call are placed alike).
    The kernels address every tensor through its TensorAccessor by logical page id, so neither the work split
    nor the kernels depend on how a tensor is sharded or how many cores its shard grid spans."""
    b_shape = () if _is_number(b) else tuple(b.shape)
    shape = list(torch.broadcast_shapes(tuple(a.shape), b_shape))
    out_dtype = dtype if dtype is not None else a.dtype
    if shape == list(a.shape) and out_dtype == a.dtype:
        return a.spec
    memory_config = a.memory_config()
    if memory_config.is_sharded() and needs_broadcast(tuple(a.shape), tuple(shape)):
        memory_config = b.memory_config()
    return ttnn.TensorSpec(ttnn.Shape(shape), out_dtype, a.layout, memory_config)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def add(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor | float,
    *,
    dtype: ttnn.DataType = None,
    memory_config: "ttnn.MemoryConfig" = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(
        input_tensor_a, input_tensor_b, dtype=dtype, memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    config = _resolve_config(compute_kernel_config)
    output = ttnn.allocate_tensor_on_device(_output_spec(input_tensor_a, input_tensor_b, dtype), input_tensor_a.device())
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor_a, input_tensor_b, output, config)
    return ttnn.generic_op(io_tensors(input_tensor_a, input_tensor_b, output), spec, run_args, tensor_indices)
