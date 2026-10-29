# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm: y = x * rsqrt(mean(x^2, dim=-1) + eps) * gamma, one generic_op dispatch per call."""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor, select_blocking

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % TILE != 0:
        return "w_non_aligned"
    if shape[-2] % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


# gamma_dtype / gamma_layout are not taggers: the golden harness does not iterate tagged axes, so a
# tagger would collapse every gamma cell. validate() reads them off the gamma tensor ("none" if absent).
INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    "gamma_layout": [ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # fp32 input requires fp32 accumulation (precision_convention.md).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, gamma=None, compute_kernel_config, memory_config=None):
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input rank must be >= 2, got shape {shape}")
    if gamma is not None:
        gamma_shape = list(gamma.shape)
        if gamma_shape[-1] != shape[-1] or math.prod(gamma_shape) != shape[-1]:
            raise ValueError(f"rms_norm: gamma shape {gamma_shape} does not match input width {shape[-1]}")

    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    if has_gamma and gamma.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("rms_norm: gamma must be interleaved")
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"rms_norm: output memory_layout={memory_config.memory_layout!r} not in SUPPORTED "
            f"{SUPPORTED['memory_layout']}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config, memory_config=memory_config)

    blocking = select_blocking(input_tensor, gamma)
    if blocking is None:
        raise NotImplementedError("rms_norm: row-resident footprint exceeds L1; regime R2 w_split_combine not built")

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config if memory_config is not None else input_tensor.memory_config(),
    )

    program_descriptor = create_program_descriptor(
        input_tensor, gamma, output_tensor, blocking, epsilon=epsilon, compute_kernel_config=compute_kernel_config
    )
    io_tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io_tensors, program_descriptor)
