# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm: layer normalisation over the last dimension with gamma and beta, on Quasar.

The specification is eval/specs/layer_norm/spec.py; the design is op_design.md in this directory. One public
call is one `ttnn.generic_op` dispatch of one Metal 2.0 `ttnn.ProgramSpec` (layer_norm_program_artifacts.py).
"""

from __future__ import annotations

import math

import ttnn

from eval.memory import memory_axes
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .layer_norm_program_artifacts import create_program_artifacts, fits_resident, resolve_compute_config


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

from eval.specs.layer_norm.spec import AXES  # noqa: E402

INPUT_TAGGERS = AXES


# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (the specification's PHASE0), axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True, False],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM"],
    "sharding":         ["interleaved"],
    "affine_dtype":     [ttnn.bfloat16],
    "affine_layout":    [ttnn.TILE_LAYOUT],
    "alignment":        ["tile_aligned"],
    "rank":             [4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none inside the first-phase rectangle.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R2): HiFi4, no approximation, fp32 accumulation."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _config_or_default(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


def _check_affine_tensors(input_tensor, weight, bias):
    """Malformed calls (REJECTIONS X1, X2; rule R9's bias-vs-weight check): ValueError."""
    width = input_tensor.shape[-1]
    for name, t in (("weight", weight), ("bias", bias)):
        if t.shape[-1] != width:
            raise ValueError(f"layer_norm: the last dimension of `{name}` is {t.shape[-1]}, the input's W is {width}")
        rows = math.prod(t.shape) // t.shape[-1]
        if rows > 32:
            raise ValueError(f"layer_norm: `{name}` has {rows} rows; it must be one tile high (at most 32 rows)")
    if bias.dtype != weight.dtype or bias.layout != weight.layout:
        raise ValueError(
            f"layer_norm: `bias` ({bias.dtype}, {bias.layout}) must have the dtype and layout of `weight` "
            f"({weight.dtype}, {weight.layout})")


def validate(input_tensor, *, epsilon=1e-12, weight, bias, compute_kernel_config=None):
    config = _config_or_default(compute_kernel_config)
    # Rule R3: neither math_fidelity nor math_approx_mode is read here.
    axes = {
        "dtype":            input_tensor.dtype,
        "layout":           input_tensor.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        **memory_axes(input_tensor),                      # rule R13
        "affine_dtype":     weight.dtype,                 # rule R9: read off the weight
        "affine_layout":    weight.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    # Rule R1: a float32 input with the 16-bit accumulator is legal but lossy, and refused.
    if axes["dtype"] == ttnn.float32 and not axes["fp32_dest_acc_en"]:
        raise UnsupportedAxisValue("layer_norm: a float32 input requires fp32_dest_acc_en=True")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"layer_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"layer_norm: unsupported combination (refinement candidate): {exc}")

    _check_affine_tensors(input_tensor, weight, bias)

    # The built regime keeps a whole row resident on one NEO; a row too wide for that is the deferred
    # `row_streaming` regime (op_design.md, Regimes).
    if not fits_resident(input_tensor, weight, bool(config.fp32_dest_acc_en)):
        raise NotImplementedError(
            f"layer_norm: W={input_tensor.shape[-1]} does not fit one NEO's L1 with the row resident "
            "(the streaming regime is not built yet)")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-12,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """(x - mean) / sqrt(var + epsilon) * gamma + beta over the last dimension, gamma = weight[0, 0, :] and
    beta = bias[0, 0, :]; one device program."""
    validate(input_tensor, epsilon=epsilon, weight=weight, bias=bias, compute_kernel_config=compute_kernel_config)
    config = resolve_compute_config(_config_or_default(compute_kernel_config))  # rule R4
    # Rule R12: the input's shape, dtype, layout and buffer type, interleaved.
    output = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, weight, bias, output, epsilon=float(epsilon), config=config
    )
    tensors = [None] * len(tensor_indices)
    for name, tensor in (("x", input_tensor), ("weight", weight), ("bias", bias), ("out", output)):
        tensors[tensor_indices[name]] = tensor
    return ttnn.generic_op(tensors, spec, run_args, tensor_indices)
