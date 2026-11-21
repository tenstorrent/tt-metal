# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""multiply: `out = a * b` with torch broadcasting, as one `ttnn.generic_op` dispatch of one Metal 2.0
`ttnn.ProgramSpec` (reader / compute / writer on every cluster that gets work).

Specification: eval/specs/multiply/spec.py. Design: op_design.md (this directory).
"""

from __future__ import annotations

import torch

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.multiply.spec import AXES, canonical_activation, canonical_memory

from .multiply_program_artifacts import create_program_artifacts, width_shards

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0 = the specification's PHASE0; Refinement 0 added memory_argument, a_activation and
#    width sharding
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved", "width"],
    "memory_argument":  [None, "input"],
    "a_activation":     [None, "silu"],
    "alignment":        ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":             [2, 3, 4],
    "broadcast":        ["none", "over_h"],
    "output_dtype":     [None, ttnn.bfloat16],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R3): the one place the default is written."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


def _axes(input_tensor_a, input_tensor_b, dtype, memory_config, input_tensor_a_activations, compute_kernel_config):
    a_shape, b_shape = tuple(input_tensor_a.shape), tuple(input_tensor_b.shape)
    try:
        torch.broadcast_shapes(a_shape, b_shape)
    except RuntimeError as e:  # X1
        raise ValueError(f"multiply: shapes {list(a_shape)} and {list(b_shape)} do not broadcast") from e
    if input_tensor_b.dtype != input_tensor_a.dtype:
        raise UnsupportedAxisValue(
            f"multiply: operands of different dtypes ({input_tensor_a.dtype}, {input_tensor_b.dtype})"
        )
    if input_tensor_b.layout != input_tensor_a.layout:
        raise UnsupportedAxisValue(
            f"multiply: operands of different layouts ({input_tensor_a.layout}, {input_tensor_b.layout})"
        )
    placed_a, placed_b = memory_axes(input_tensor_a), memory_axes(input_tensor_b)  # R10
    if placed_a != placed_b:
        raise UnsupportedAxisValue(f"multiply: operands placed differently ({placed_a}, {placed_b})")
    config = _resolve_config(compute_kernel_config)
    axes = {
        "dtype": input_tensor_a.dtype,
        "layout": input_tensor_a.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "memory_config": placed_a["memory_config"],
        "sharding": placed_a["sharding"],
        "output_dtype": dtype,
        # The specification's canonical readings (CALL["canonical"]): anything but None / the first operand's
        # own config, and None / [SILU], reads as "other", which no SUPPORTED list holds.
        "memory_argument": canonical_memory(memory_config, {"input_tensor_a": input_tensor_a}),
        "a_activation": canonical_activation(input_tensor_a_activations, {}),
    }
    inputs = (list(a_shape), list(b_shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(
    input_tensor_a,
    input_tensor_b,
    *,
    dtype=None,
    memory_config=None,
    input_tensor_a_activations=None,
    compute_kernel_config=None,
):
    axes = _axes(input_tensor_a, input_tensor_b, dtype, memory_config, input_tensor_a_activations, compute_kernel_config)

    # R1 / R2: a 16-bit accumulator cannot give a float32 result. (R4: math_fidelity and math_approx_mode are
    # never read here.)
    if not axes["fp32_dest_acc_en"]:
        if axes["dtype"] == ttnn.float32:
            raise UnsupportedAxisValue("multiply: float32 operands need fp32_dest_acc_en=True (R1)")
        if axes["output_dtype"] == ttnn.float32:
            raise UnsupportedAxisValue("multiply: dtype=float32 needs fp32_dest_acc_en=True (R2)")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"multiply: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"multiply: unsupported combination (refinement candidate): {exc}")

    if axes["sharding"] == "width":
        try:  # the two operands' shards must cover the same columns on the same cores
            width_shards(input_tensor_a, input_tensor_b)
        except NotImplementedError as e:
            raise UnsupportedAxisValue(str(e)) from e
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _allocate_output(input_tensor_a, input_tensor_b, dtype):
    out_shape = list(torch.broadcast_shapes(tuple(input_tensor_a.shape), tuple(input_tensor_b.shape)))
    out_dtype = dtype if dtype is not None else input_tensor_a.dtype
    # R9: the first operand's placement — interleaved in its buffer, or width-sharded like it. `memory_config=`
    # can only name that same config (validate()), so it needs no branch here.
    spec = ttnn.TensorSpec(ttnn.Shape(out_shape), out_dtype, input_tensor_a.layout, input_tensor_a.memory_config())
    return ttnn.allocate_tensor_on_device(spec, input_tensor_a.device())


def _dispatch(input_tensor_a, input_tensor_b, axes, compute_kernel_config, **artifact_options):
    out = _allocate_output(input_tensor_a, input_tensor_b, axes["output_dtype"])
    spec, run_args, tensor_indices, plan = create_program_artifacts(
        input_tensor_a,
        input_tensor_b,
        out,
        broadcast=axes["broadcast"],
        compute_kernel_config=_resolve_config(compute_kernel_config),
        sharding=axes["sharding"],
        a_activation=axes["a_activation"],
        **artifact_options,
    )
    return ttnn.generic_op([input_tensor_a, input_tensor_b, out], spec, run_args, tensor_indices), plan


def multiply(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    dtype: ttnn.DataType = None,
    memory_config: "ttnn.MemoryConfig" = None,
    input_tensor_a_activations=None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    axes = validate(
        input_tensor_a,
        input_tensor_b,
        dtype=dtype,
        memory_config=memory_config,
        input_tensor_a_activations=input_tensor_a_activations,
        compute_kernel_config=compute_kernel_config,
    )
    output, _ = _dispatch(input_tensor_a, input_tensor_b, axes, compute_kernel_config)
    return output
